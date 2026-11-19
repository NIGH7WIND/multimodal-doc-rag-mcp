import os
import re
import time
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright

BASE_URL = "https://docs.example.com/getting-started"
DOMAIN = "https://docs.example.com"
OUTPUT_DIR = Path("rag_repository")
USER_DATA_DIR = Path("./chrome_profile")  # Saves Google login session


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:100] or "untitled"


def expand_all_sidebar_items(page):
    """Repeatedly clicks any button with aria-label='Expand' until all nested menus are open."""
    while True:
        expand_buttons = page.locator('[data-sidebar="menu"] button[aria-label="Expand"]')
        count = expand_buttons.count()
        if count == 0:
            break
        for i in range(count):
            try:
                btn = expand_buttons.nth(i)
                if btn.is_visible():
                    btn.click()
                    time.sleep(0.3)
            except Exception:
                pass
        time.sleep(0.5)


def collect_sidebar_links(page) -> list[dict]:
    """Extracts all unique hrefs and titles from the expanded sidebar."""
    links = []
    seen_urls = set()
    anchors = page.locator('[data-sidebar="menu"] a[href]')
    
    for i in range(anchors.count()):
        a = anchors.nth(i)
        href = a.get_attribute("href")
        title = a.inner_text().strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue
        full_url = urllib.parse.urljoin(DOMAIN, href)
        if full_url not in seen_urls and full_url.startswith(DOMAIN):
            seen_urls.add(full_url)
            links.append({"title": title, "url": full_url, "path": href})
    return links


def download_media(page, media_url: str, save_path: Path) -> bool:
    """Downloads images/videos using Playwright's authenticated APIRequestContext."""
    try:
        if media_url.startswith("blob:"):
            # Handle blob URLs via browser fetch -> base64
            b64_data = page.evaluate(
                """async (url) => {
                    const res = await fetch(url);
                    const blob = await res.blob();
                    return new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result.split(',')[1]);
                        reader.readAsDataURL(blob);
                    });
                }""",
                media_url,
            )
            import base64
            save_path.write_bytes(base64.b64decode(b64_data))
            return True
        else:
            full_url = urllib.parse.urljoin(DOMAIN, media_url)
            response = page.request.get(full_url, timeout=60000)
            if response.ok:
                save_path.write_bytes(response.body())
                return True
    except Exception as e:
        print(f"    [!] Failed to download {media_url}: {e}")
    return False


def scrape_page(page, item: dict, index: int):
    url = item["url"]
    title = item["title"]
    slug = sanitize_filename(f"{index:02d}_{title}")
    page_dir = OUTPUT_DIR / slug
    images_dir = page_dir / "images"
    videos_dir = page_dir / "videos"

    print(f"\n[{index:02d}] Scraping: {title} ({url})")
    page.goto(url, wait_until="networkidle", timeout=60000)
    time.sleep(1.5)  # Allow lazy-loaded media/hydrated components to settle

    # Locate main content area (fallback to body if <main> isn't used)
    main_locator = page.locator("main").first
    if main_locator.count() == 0:
        main_locator = page.locator("body")

    raw_html = main_locator.inner_html()
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove UI noise (copy buttons, right-hand TOC if inside main, scripts/styles)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 1. Extract & Download Images
    img_tags = soup.find_all("img")
    if img_tags:
        images_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(img_tags, start=1):
        src = img.get("src") or img.get("data-src")
        if not src or src.startswith("data:image/svg"):
            continue
        ext = os.path.splitext(urllib.parse.urlparse(src).path)[1] or ".png"
        if len(ext) > 5:
            ext = ".png"
        img_filename = f"image_{idx:02d}{ext}"
        if download_media(page, src, images_dir / img_filename):
            img["src"] = f"./images/{img_filename}"
            print(f"    [+] Saved image: {img_filename}")

    # 2. Extract & Download Videos
    video_tags = soup.find_all("video")
    video_notes = []
    if video_tags:
        videos_dir.mkdir(parents=True, exist_ok=True)
    for idx, vid in enumerate(video_tags, start=1):
        src = vid.get("src")
        if not src:
            source_tag = vid.find("source")
            if source_tag:
                src = source_tag.get("src")
        if src:
            ext = os.path.splitext(urllib.parse.urlparse(src).path)[1] or ".mp4"
            if len(ext) > 5:
                ext = ".mp4"
            vid_filename = f"video_{idx:02d}{ext}"
            if download_media(page, src, videos_dir / vid_filename):
                print(f"    [+] Saved video: {vid_filename}")
                # Replace <video> tag with a clean Markdown-friendly placeholder
                placeholder = soup.new_tag("p")
                placeholder.string = f"[Local Video: ./videos/{vid_filename}]"
                vid.replace_with(placeholder)
                video_notes.append(f"./videos/{vid_filename}")

    # 3. Convert cleaned HTML to Markdown
    page_dir.mkdir(parents=True, exist_ok=True)
    markdown_body = md(
        str(soup),
        heading_style="ATX",
        bullets="-",
        code_language_callback=lambda el: el.get("data-language", "") if el else "",
    )
    # Clean up excessive blank lines
    markdown_body = re.sub(r"\n{3,}", "\n\n", markdown_body).strip()

    # Add YAML Frontmatter for RAG metadata filtering
    frontmatter = [
        "---",
        f'title: "{title.replace(chr(34), "")}"',
        f'url: "{url}"',
        f'path: "{item["path"]}"',
        f"videos: {video_notes}",
        "---\n\n",
    ]

    md_file = page_dir / f"{slug}.md"
    md_file.write_text("\n".join(frontmatter) + markdown_body, encoding="utf-8")
    print(f"    [âœ“] Saved Markdown: {md_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Use system Chrome + persistent context so Google SSO doesn't block login
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.new_page()
        page.goto(BASE_URL)

        print("\n" + "=" * 60)
        print("ACTION REQUIRED:")
        print("1. Log in with your Google account in the opened Chrome window.")
        print("2. Once you see the 'Getting started' page and left sidebar,")
        print("   come back here and press ENTER to start scraping.")
        print("=" * 60)
        input()

        print("[*] Expanding all collapsible sidebar items...")
        expand_all_sidebar_items(page)

        links = collect_sidebar_links(page)
        print(f"[*] Found {len(links)} pages in the sidebar.")

        for idx, item in enumerate(links, start=1):
            try:
                scrape_page(page, item, idx)
            except Exception as e:
                print(f"[!] Error scraping {item['url']}: {e}")

        print(f"\n[DONE] All content saved to: {OUTPUT_DIR.resolve()}")
        context.close()


if __name__ == "__main__":
    main()
