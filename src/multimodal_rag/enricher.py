# === 1. MARKDOWN CLEANER ===
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml",
# ]
# ///

import re
from pathlib import Path
import yaml

RAW_DIR = Path("rag_repository")
CLEAN_DIR = Path("rag_repository_clean")

# UI boilerplate lines that add zero semantic value
NOISE_LINES = {
    "Project Insight",
    "Are you onboarding?",
    "Scroll through this training page and then return to the Platform to task.",
    "Before you get started",
    "Copy",
    "Copied",
}


def strip_bottom_toc_and_nav(text: str) -> str:
    """
    Removes the trailing right-hand Table of Contents (- [Title](#slug))
    and Prev/Next page navigation links at the bottom of the document.
    """
    lines = text.splitlines()
    # Walk backwards from the end and strip trailing TOC anchor lists & prev/next link bars
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        # Matches trailing TOC bullet: - [Section Name](#section-name)
        if re.fullmatch(r"-\s+\[[^\]]+\]\(#[^)]+\)", last):
            lines.pop()
            continue
        # Matches trailing Prev/Next nav links: [Prev](/a)[Next](/b) or [Next](/b)
        if re.fullmatch(r"(\[[^\]]+\]\(/[^)]+\)\s*)+", last):
            lines.pop()
            continue
        break
    return "\n".join(lines)


def fix_numbered_step_headings(text: str) -> str:
    """
    Merges orphaned badge numbers above headings:
    '1\n\n#### Proposal' -> '#### 1. Proposal'
    """
    return re.sub(
        r"(?m)^(\d+)\s*\n+(#{2,6})\s+(.+)$",
        r"\2 \1. \3",
        text,
    )


def format_pricing_cards(text: str) -> str:
    """
    Converts 3-part vertical pricing cards into clean bullet points:
    'United States\n\n$600\n\nPer task, after RTD clears'
    -> '- **United States:** $600 â€” Per task, after RTD clears'
    """
    return re.sub(
        r"(?m)^([A-Z][A-Za-z\s]+)\n+(\$\d[\d,]*(?:/\w+)?)\n+([^\n]+)$",
        r"- **\1:** \2 â€” \3",
        text,
    )


def format_callout_blocks(text: str) -> str:
    """
    Converts standalone short callout titles followed by a paragraph into blockquotes:
    '\\*Revise sent-back tasks before starting new ones\n\nIf R1 sent...'
    or 'Off-platform work\n\nThe project runs...'
    """
    # 1. Escaped asterisk callout headings (\*Title)
    text = re.sub(
        r"(?m)^\\?\*([^\n]+)\n+([^\n#\-*>].+)$",
        r"> **\1**\n> \2",
        text,
    )

    # 2. Known short UI callout titles without punctuation followed by a body paragraph
    callout_titles = [
        "Off-platform work",
        "Note",
        "Tip",
        "Important",
        "Warning",
    ]
    for title in callout_titles:
        text = re.sub(
            rf"(?m)^{re.escape(title)}\n+([^\n#\-*>].+)$",
            rf"> **{title}:** \1",
            text,
        )
    return text


def clean_card_links(text: str) -> str:
    """Fixes jammed React card links and adjacent buttons."""
    # 1. Multi-line card links: [### Heading\n\nBody](/url)
    def unwrap_multiline_link(match):
        inner = re.sub(r"\n{2,}", "\n", match.group(1).strip())
        url = match.group(2).strip()
        return f"{inner} ([Reference]({url}))"

    text = re.sub(r"\[([^\]]+\n[^\]]+)\]\(([^)]+)\)", unwrap_multiline_link, text)

    # 2. Numbered discovery cards: [01Full flow at a glanceThe eight steps...7 min](/read-this-first)
    def format_numbered_card(match):
        num = match.group(1)
        body = re.sub(r"([a-z])([A-Z])", r"\1 â€” \2", match.group(2))
        duration = f" ({match.group(3)})" if match.group(3) else ""
        url = match.group(4)
        return f"- **{num}. {body}**{duration} â†’ `{url}`"

    text = re.sub(
        r"\[(\d{2})([^\]]+?)(\d+\s*min)?\]\((/[^)]+)\)",
        format_numbered_card,
        text,
    )

    # 3. Separate adjacent inline buttons: [A](/a)[B](/b) -> [A](/a) | [B](/b)
    text = re.sub(r"\]\(([^)]+)\)\[", r"](\1) | [", text)
    return text


def merge_stat_blocks(text: str) -> str:
    """
    Merges orphaned stat numbers with their description line:
    '8\n\nSteps from claiming...' -> '- **8** â€” Steps from claiming...'
    """
    return re.sub(
        r"(?m)^(\d+)\s*\n+([A-Z][^\n#]+)$",
        r"- **\1** â€” \2",
        text,
    )


def clean_footnotes(text: str) -> str:
    """Formats inline footnote definitions cleanly: '1.Measured' -> '[^1]: Measured'"""
    text = re.sub(r"(?m)^(\d+)\.([A-Z])", r"[^\1]: \2", text)
    return text


def remove_ui_noise(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in NOISE_LINES:
            continue
        # Remove standalone internal CTA button lines like "[Get started](/getting-started)"
        if re.fullmatch(r"\[[^\]]+\]\(/[^)]+\)(\s*\|\s*\[[^\]]+\]\(/[^)]+\))*", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def inject_context_header(content: str, meta: dict) -> str:
    title = meta.get("title", "Untitled")
    path = meta.get("path", "")
    if content.lstrip().startswith("# "):
        return content
    return f"# {title} (`{path}`)\n\n" + content


def process_all():
    for md_path in sorted(RAW_DIR.rglob("*.md")):
        raw_text = md_path.read_text(encoding="utf-8")

        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
        else:
            meta = {}
            body = raw_text

        body = strip_bottom_toc_and_nav(body)
        body = fix_numbered_step_headings(body)
        body = format_pricing_cards(body)
        body = format_callout_blocks(body)
        body = clean_card_links(body)
        body = merge_stat_blocks(body)
        body = clean_footnotes(body)
        body = remove_ui_noise(body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        body = inject_context_header(body, meta)

        out_path = CLEAN_DIR / md_path.relative_to(RAW_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = yaml.dump(meta, sort_keys=False).strip()
        out_path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
        print(f"[âœ“] Cleaned: {out_path}")


if __name__ == "__main__":
    process_all()

# === 2. VLM IMAGE CAPTIONER ===
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "pillow",
# ]
# ///

import base64
import io
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from openai import OpenAI
from PIL import Image

RAW_DIR = Path("rag_repository")
CLEAN_DIR = Path("rag_repository_clean")

LLAMA_SERVER_URL = "http://localhost:8080/v1"
MODEL_NAME = "gemma-4"

client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")

# Tags inside SVGs whose text content is CSS/code rather than visible labels
SVG_IGNORED_TAGS = {"style", "script", "defs", "clippath", "mask", "pattern"}


def repair_mangled_images(text: str) -> str:
    """Repairs mangled clickable image syntax from clean_markdown.py."""
    pattern = re.compile(
        r"!([^\n]+)\n+(?:</?\d+>\s*)?\(\[Reference\]\((\./images/[^)]+)\)\)"
    )
    return pattern.sub(r"![\1](\2)", text)


def extract_clean_svg_content(svg_path: Path) -> str:
    """
    Extracts readable <title>, <desc>, and <text>/<tspan> labels from an SVG
    while completely ignoring <style> CSS blocks and <defs>.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        def walk(node):
            # Strip XML namespace prefix e.g. '{http://www.w3.org/2000/svg}style' -> 'style'
            tag_name = node.tag.split("}")[-1].lower()
            if tag_name in SVG_IGNORED_TAGS:
                return []

            collected = []
            if node.text and node.text.strip():
                collected.append(node.text.strip())
            for child in node:
                collected.extend(walk(child))
                if child.tail and child.tail.strip():
                    collected.append(child.tail.strip())
            return collected

        raw_pieces = walk(root)
        # Deduplicate preserving order
        seen = set()
        clean_pieces = [p for p in raw_pieces if not (p in seen or seen.add(p))]
        return " | ".join(clean_pieces)
    except Exception:
        return ""


def image_to_base64_png(img_path: Path) -> str | None:
    """Converts PNG/JPG/WEBP to base64 PNG string for llama-server."""
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"    [!] Could not read image {img_path}: {e}")
        return None


def generate_caption(img_path: Path, alt_text: str, surrounding_context: str) -> str:
    """Queries local Gemma 4 on llama.cpp to produce a clean RAG caption."""
    base_instructions = (
        "You are indexing technical documentation for a RAG knowledge base.\n"
        f"Document context around this visual:\n---\n{surrounding_context}\n---\n"
        f"Alt text: {alt_text or 'None'}\n\n"
        "Write a clear, natural 2-4 sentence description so a developer searching via text "
        "can find its exact contents. Include key labels, states, button names, scores, "
        "or workflow logic. Do not use markdown headings, bullet lists, raw CSS, or filler "
        "phrases like 'This image shows'."
    )

    try:
        if img_path.suffix.lower() == ".svg":
            svg_labels = extract_clean_svg_content(img_path)
            if not svg_labels:
                return alt_text or "Vector diagram."

            # Ask Gemma 4 (text mode) to turn the clean SVG text nodes into a readable summary
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.2,
                max_tokens=250,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{base_instructions}\n\n"
                            f"Extracted diagram labels & accessibility text:\n{svg_labels}"
                        ),
                    }
                ],
            )
        else:
            b64_img = image_to_base64_png(img_path)
            if not b64_img:
                return alt_text or "Image could not be processed."

            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.2,
                max_tokens=250,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": base_instructions},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                            },
                        ],
                    }
                ],
            )

        caption = response.choices[0].message.content.strip()
        return re.sub(r"\s+", " ", caption)
    except Exception as e:
        print(f"    [!] llama-server error on {img_path.name}: {e}")
        return alt_text or "Caption generation failed."


def process_markdown_file(md_path: Path, force: bool = False):
    rel_folder = md_path.parent.relative_to(CLEAN_DIR)
    raw_images_dir = RAW_DIR / rel_folder / "images"
    clean_images_dir = md_path.parent / "images"

    if raw_images_dir.exists() and not clean_images_dir.exists():
        shutil.copytree(raw_images_dir, clean_images_dir)

    text = md_path.read_text(encoding="utf-8")
    text = repair_mangled_images(text)

    if force:
        # Remove existing [Caption: ...] blocks so we can regenerate cleanly
        text = re.sub(r"\n+\[Caption:[^\]]*\]", "", text)

    img_pattern = re.compile(r"(!\[([^\]]*)\]\((\./images/[^)]+)\))(?!\n+\[Caption:)")
    matches = list(img_pattern.finditer(text))
    if not matches:
        md_path.write_text(text, encoding="utf-8")
        return

    print(f"\n[*] Processing {len(matches)} image(s) in {md_path.name}...")

    for match in reversed(matches):
        full_md_img = match.group(1)
        alt_text = match.group(2).strip()
        rel_img_path = match.group(3).strip()

        img_file = (md_path.parent / rel_img_path).resolve()
        if not img_file.exists():
            print(f"    [!] Missing file: {img_file}")
            continue

        start_ctx = max(0, match.start() - 300)
        end_ctx = min(len(text), match.end() + 300)
        context_snippet = text[start_ctx:end_ctx].strip()

        print(f"    -> Captioning {rel_img_path}...")
        caption = generate_caption(img_file, alt_text, context_snippet)

        replacement = f"{full_md_img}\n[Caption: {caption}]"
        text = text[: match.start()] + replacement + text[match.end() :]
        print(f"    [+] Added: [Caption: {caption[:90]}...]")

    md_path.write_text(text, encoding="utf-8")


def main():
    force = "--force" in sys.argv
    for md_path in sorted(CLEAN_DIR.rglob("*.md")):
        process_markdown_file(md_path, force=force)
    print("\n[DONE] All images captioned and embedded into clean Markdown files.")


if __name__ == "__main__":
    main()

# === 3. WHISPER VIDEO TRANSCRIBER ===
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "faster-whisper",
#     "av<14",
#     "nvidia-cublas-cu12",
#     "nvidia-cudnn-cu12",
#     "pyyaml",
# ]
# ///

import os
import re
import shutil
import site
import sys
from pathlib import Path
import yaml

os.environ["HF_HUB_DISABLE_XET"] = "1"


def register_windows_cuda_dlls():
    """Registers uv's nvidia cublas/cudnn bin directories so Windows finds them immediately."""
    if sys.platform != "win32":
        return
    for sp in site.getsitepackages():
        nvidia_dir = Path(sp) / "nvidia"
        if not nvidia_dir.exists():
            continue
        for sub in ("cublas", "cudnn"):
            bin_dir = nvidia_dir / sub / "bin"
            if bin_dir.exists():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


register_windows_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402

RAW_DIR = Path("rag_repository")
CLEAN_DIR = Path("rag_repository_clean")

# distil-large-v3 is ~756MB (half the size of large-v3-turbo) and fast on RTX 3060
MODEL_SIZE = "distil-large-v3"


def load_whisper_model() -> WhisperModel:
    try:
        print(f"[*] Downloading/loading '{MODEL_SIZE}' on CUDA (float16)...", flush=True)
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        print("[âœ“] Model loaded on GPU.", flush=True)
        return model
    except Exception as e:
        print(f"[!] CUDA failed ({e}), falling back to CPU (int8)...", flush=True)
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def transcribe_video(model: WhisperModel, video_path: Path, page_title: str) -> str:
    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=(
            f"Technical tutorial walkthrough for {page_title}. "
            "Topics include Terminal-Bench, Harbor, Docker, GitHub pull requests, "
            "task.toml, instruction.md, pass@2, and verifier tests."
        ),
    )

    paragraphs = []
    current_chunk = []
    chunk_start = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if chunk_start is None:
            chunk_start = seg.start
        current_chunk.append(text)

        if (seg.end - chunk_start) >= 35.0:
            ts = format_timestamp(chunk_start)
            paragraphs.append(f"[{ts}] {' '.join(current_chunk)}")
            current_chunk = []
            chunk_start = None

    if current_chunk and chunk_start is not None:
        ts = format_timestamp(chunk_start)
        paragraphs.append(f"[{ts}] {' '.join(current_chunk)}")

    return "\n".join(paragraphs) if paragraphs else "[No spoken audio detected]"


def process_markdown_file(model: WhisperModel, md_path: Path, force: bool = False):
    rel_folder = md_path.parent.relative_to(CLEAN_DIR)
    raw_videos_dir = RAW_DIR / rel_folder / "videos"
    clean_videos_dir = md_path.parent / "videos"

    if raw_videos_dir.exists() and not clean_videos_dir.exists():
        shutil.copytree(raw_videos_dir, clean_videos_dir)

    raw_text = md_path.read_text(encoding="utf-8")

    page_title = md_path.stem
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        meta = yaml.safe_load(parts[1]) or {}
        page_title = meta.get("title", page_title)

    if force:
        raw_text = re.sub(r"\n+\[Video Transcript:[\s\S]*?\n\]", "", raw_text)

    vid_pattern = re.compile(
        r"(\[Local Video:\s*(\./videos/[^\]]+)\])(?!\n+\[Video Transcript:)"
    )
    matches = list(vid_pattern.finditer(raw_text))

    if not matches:
        return

    print(f"\n[*] Found {len(matches)} video(s) in {md_path.name}...", flush=True)

    for match in reversed(matches):
        full_marker = match.group(1)
        rel_vid_path = match.group(2).strip().replace("\\_", "_")
        vid_file = (md_path.parent / rel_vid_path).resolve()

        if not vid_file.exists():
            print(f"    [!] Missing video file: {vid_file}", flush=True)
            continue

        print(f"    -> Transcribing {rel_vid_path}...", flush=True)
        transcript = transcribe_video(model, vid_file, page_title)

        block = f"{full_marker}\n[Video Transcript:\n{transcript}\n]"
        raw_text = raw_text[: match.start()] + block + raw_text[match.end() :]
        print(f"    [+] Embedded transcript ({len(transcript)} chars)", flush=True)

    md_path.write_text(raw_text, encoding="utf-8")


def main():
    force = "--force" in sys.argv
    model = load_whisper_model()

    for md_path in sorted(CLEAN_DIR.rglob("*.md")):
        process_markdown_file(model, md_path, force=force)

    print("\n[DONE] All video transcripts embedded into clean Markdown files.", flush=True)


if __name__ == "__main__":
    main()

