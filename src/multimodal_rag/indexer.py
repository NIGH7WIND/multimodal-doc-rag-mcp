import os
from dotenv import load_dotenv
load_dotenv()
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lancedb",
#     "fastembed",
#     "pyyaml",
#     "tantivy",
# ]
# ///

import re
import sys
from pathlib import Path
import lancedb
import yaml
from fastembed import TextEmbedding

CLEAN_DIR = Path("rag_repository_clean")
DB_DIR = Path("lancedb_store")
TABLE_NAME = os.getenv("LANCEDB_TABLE", "technical_docs")
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

MAX_CHUNK_CHARS = 2400  # ~600 tokens
OVERLAP_CHARS = 300


def parse_markdown_sections(md_path: Path) -> list[dict]:
    """Splits a Markdown file by headers while preserving frontmatter and breadcrumbs."""
    raw = md_path.read_text(encoding="utf-8")
    meta = {}
    body = raw

    if raw.startswith("---"):
        parts = raw.split("---", 2)
        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()

    doc_title = meta.get("title", md_path.stem)
    doc_url = meta.get("url", "")
    doc_path = meta.get("path", "")

    lines = body.splitlines()
    sections = []
    h1, h2, h3 = doc_title, "", ""
    buf = []
    in_code_block = False

    def flush():
        text = "\n".join(buf).strip()
        if not text:
            return
        breadcrumb = " > ".join([p for p in (h1, h2, h3) if p])
        sections.append(
            {
                "doc_title": doc_title,
                "url": doc_url,
                "path": doc_path,
                "breadcrumb": breadcrumb,
                "text": text,
            }
        )

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        if not in_code_block:
            if line.startswith("# "):
                flush()
                buf = []
                h1 = line[2:].strip()
                h2, h3 = "", ""
                continue
            elif line.startswith("## "):
                flush()
                buf = []
                h2 = line[3:].strip()
                h3 = ""
                continue
            elif line.startswith("### "):
                flush()
                buf = []
                h3 = line[4:].strip()
                continue

        buf.append(line)

    flush()
    return sections


def split_large_section(sec: dict) -> list[dict]:
    """Sub-splits oversized sections on paragraph boundaries without breaking code blocks."""
    text = sec["text"]
    if len(text) <= MAX_CHUNK_CHARS:
        return [sec]

    blocks = re.split(r"(\n```[\s\S]*?```\n|\n\n+)", text)
    chunks = []
    current = ""

    for blk in blocks:
        if not blk or not blk.strip():
            continue
        if len(current) + len(blk) <= MAX_CHUNK_CHARS:
            current += ("\n\n" if current else "") + blk.strip()
        else:
            if current:
                chunks.append({**sec, "text": current})
            tail = current[-OVERLAP_CHARS:] if current and not current.endswith("```") else ""
            current = (tail + "\n\n" + blk.strip()).strip() if tail else blk.strip()

    if current:
        chunks.append({**sec, "text": current})
    return chunks


def build_index():
    print(f"[*] Loading ONNX embedding model: {EMBED_MODEL_NAME}...", flush=True)
    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)

    records = []
    for md_path in sorted(CLEAN_DIR.rglob("*.md")):
        sections = parse_markdown_sections(md_path)
        for sec in sections:
            for chunk in split_large_section(sec):
                contextualized = (
                    f"Document: {chunk['doc_title']} | "
                    f"Section: {chunk['breadcrumb']} | "
                    f"Source: {chunk['path']}\n\n{chunk['text']}"
                )
                records.append(
                    {
                        "id": f"{md_path.stem}_{len(records)}",
                        "doc_title": chunk["doc_title"],
                        "breadcrumb": chunk["breadcrumb"],
                        "url": chunk["url"],
                        "path": chunk["path"],
                        "text": contextualized,
                    }
                )

    print(f"[*] Encoding {len(records)} chunks...", flush=True)
    vectors = list(embedder.embed([r["text"] for r in records]))
    for rec, vec in zip(records, vectors):
        rec["vector"] = vec.tolist()

    db = lancedb.connect(str(DB_DIR))
    table = db.create_table(TABLE_NAME, data=records, mode="overwrite")
    table.create_fts_index("text", replace=True)
    print(f"[âœ“] Indexed {len(records)} chunks into {DB_DIR}/{TABLE_NAME}", flush=True)


def search(query: str, top_k: int = 5):
    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    db = lancedb.connect(str(DB_DIR))
    table = db.open_table(TABLE_NAME)

    # query_embed() automatically applies BGE's retrieval query prefix
    q_vec = list(embedder.query_embed(query))[0].tolist()

    results = (
        table.search(query_type="hybrid")
        .vector(q_vec)
        .text(query)
        .limit(top_k)
        .to_list()
    )

    print(f"\n=== Top {len(results)} Hybrid Results for: '{query}' ===\n")
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] {r['breadcrumb']} ({r['url']})")
        print(r["text"][:450].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        q = " ".join(sys.argv[2:]) or "What is the 8-hour time tracking cap and RTD payout?"
        search(q)
    else:
        build_index()

