import os
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lancedb",
#     "openai",
# ]
# ///

import json
import random
import re
from pathlib import Path
import lancedb
from openai import OpenAI

DB_DIR = "lancedb_store"
TABLE_NAME = os.getenv("LANCEDB_TABLE", "technical_docs")
EVAL_DATASET_PATH = Path("eval_golden_set.json")

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")


def extract_modality_payload(text: str, modality: str) -> str:
    """Extracts the target modality block so Gemma 4 focuses on it specifically."""
    if modality == "image_caption":
        captions = re.findall(r"\[Caption:\s*([^\]]+)\]", text)
        return "\n".join(f"- {c.strip()}" for c in captions) if captions else text

    if modality == "video_transcript":
        transcripts = re.findall(r"\[Video Transcript:\s*([\s\S]*?)\n\]", text)
        return "\n\n".join(t.strip() for t in transcripts) if transcripts else text

    return text


def build_prompt(row: dict, modality: str) -> str:
    header = f"Document: {row['doc_title']} | Section: {row['breadcrumb']}\n\n"

    if modality == "image_caption":
        caption_only = extract_modality_payload(row["text"], "image_caption")
        return (
            "You are creating a RAG evaluation benchmark.\n"
            "Below is an image/diagram caption extracted from technical documentation.\n"
            "Write ONE specific, self-contained question that a task author would ask whose answer "
            "depends specifically on the visual details described in this caption (not generic prose).\n"
            "Rules:\n"
            "- Do NOT ask 'Where can I find...' or ask for a URL/path as the answer.\n"
            "- Make the question specific enough to point unambiguously to this topic.\n"
            'Return valid JSON with exact keys: {"question": "...", "reference_answer": "..."}\n\n'
            f"{header}Image Caption(s):\n{caption_only}"
        )

    if modality == "video_transcript":
        transcript_only = extract_modality_payload(row["text"], "video_transcript")
        return (
            "You are creating a RAG evaluation benchmark.\n"
            "Below is an audio transcript extracted from a tutorial video.\n"
            "Write ONE specific, practical question a task author would ask based ONLY on concrete "
            "instructions, rules, or explanations spoken in this transcript.\n"
            "Rules:\n"
            "- Do NOT ask 'Where can I find...' or ask for a URL/path as the answer.\n"
            "- Make the question self-contained and specific to the workflow/tool discussed.\n"
            'Return valid JSON with exact keys: {"question": "...", "reference_answer": "..."}\n\n'
            f"{header}Video Transcript:\n{transcript_only[:3000]}"
        )

    return (
        "You are creating a RAG evaluation benchmark.\n"
        "Based ONLY on the documentation chunk below, write ONE specific, practical question "
        "a task author would ask, and a concise ground-truth reference answer.\n"
        "Rules:\n"
        "- Do NOT ask 'Where can I find...' or ask for a URL/path as the answer.\n"
        "- Avoid overly broad questions (like 'What should I avoid?') that apply to many pages; "
        "anchor the question to the specific tool, file, command, or domain example in this chunk.\n"
        'Return valid JSON with exact keys: {"question": "...", "reference_answer": "..."}\n\n'
        f"Chunk:\n{row['text']}"
    )


def main():
    db = lancedb.connect(DB_DIR)
    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    # Filter out tiny/stub chunks (< 180 chars) that lack substantive facts
    df = df[df["text"].str.len() >= 180].copy()

    has_caption = df["text"].str.contains(r"\[Caption:", regex=True)
    has_whisper_transcript = df["text"].str.contains(r"\[Video Transcript:", regex=True)

    caption_pool = df[has_caption].to_dict("records")
    video_pool = df[has_whisper_transcript].to_dict("records")
    text_pool = df[~has_caption & ~has_whisper_transcript].to_dict("records")

    random.seed(42)
    seen_ids = set()
    plan = []

    def pick_unique(pool: list[dict], count: int, modality: str):
        candidates = [r for r in pool if r["id"] not in seen_ids]
        chosen = random.sample(candidates, min(count, len(candidates)))
        for item in chosen:
            seen_ids.add(item["id"])
            plan.append((item, modality))

    # Sample strictly disjoint sets: 10 text + 5 image captions + 5 Whisper video transcripts
    pick_unique(text_pool, 10, "markdown_text")
    pick_unique(caption_pool, 5, "image_caption")
    pick_unique(video_pool, 5, "video_transcript")

    golden_set = []
    print(f"[*] Generating {len(plan)} stratified, deduplicated Q&A pairs via Gemma 4...\n", flush=True)

    for idx, (row, modality) in enumerate(plan, 1):
        prompt = build_prompt(row, modality)

        res = client.chat.completions.create(
            model="gemma-4",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        qa = json.loads(res.choices[0].message.content)

        golden_set.append(
            {
                "id": f"eval_{idx:02d}",
                "chunk_id": row["id"],
                "modality": modality,
                "expected_path": row["path"],
                "expected_breadcrumb": row["breadcrumb"],
                "question": qa["question"].strip(),
                "reference_answer": qa["reference_answer"].strip(),
            }
        )
        print(f"  [{idx:02d}/{len(plan)}] ({modality}) {qa['question'].strip()}")

    EVAL_DATASET_PATH.write_text(json.dumps(golden_set, indent=2), encoding="utf-8")
    print(f"\n[âœ“] Saved {len(golden_set)} clean test cases to {EVAL_DATASET_PATH}")


if __name__ == "__main__":
    main()
