import os
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lancedb",
#     "fastembed",
#     "openai",
#     "tantivy",
# ]
# ///

import sys
import lancedb
from fastembed import TextEmbedding
from lancedb.rerankers import RRFReranker
from openai import OpenAI

DB_DIR = "lancedb_store"
TABLE_NAME = os.getenv("LANCEDB_TABLE", "technical_docs")
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", "./.fastembed_cache")

LLAMA_SERVER_URL = "http://localhost:8080/v1"
MODEL_NAME = "gemma-4"

SYSTEM_PROMPT = """You are a precise technical assistant for technical documentation and benchmark authoring.
Answer the user's question using ONLY the provided reference context below.
Rules:
1. Cite the exact source path (e.g., ``) for every claim or rule.
2. Preserve exact CLI commands, file names (`task.toml`, `instruction.md`), and thresholds (`pass@2`, `36-72 hours`, `8-hour cap`) verbatim.
3. If the provided context does not contain the answer, state clearly that it is not covered in the indexed documentation."""


def retrieve_context(embedder: TextEmbedding, table, query: str, top_k: int = 5) -> list[dict]:
    q_vec = list(embedder.query_embed(query))[0].tolist()
    reranker = RRFReranker()
    return (
        table.search(query_type="hybrid")
        .vector(q_vec)
        .text(query)
        .rerank(reranker=reranker)
        .limit(top_k)
        .to_list()
    )


def ask_rag(client: OpenAI, embedder: TextEmbedding, table, question: str):
    chunks = retrieve_context(embedder, table, question, top_k=5)

    formatted_context = "\n\n---\n\n".join(
        f"[{i}] {c['breadcrumb']} (URL: {c['url']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Reference Context:\n\n{formatted_context}\n\n---\n\nQuestion: {question}",
        },
    ]

    print("\nSources retrieved:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['breadcrumb']} -> {c['path']}")

    print("\nAnswer:\n", flush=True)
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n")


def main():
    print("[*] Connecting to LanceDB and loading BGE embedder...", flush=True)
    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=CACHE_DIR)
    db = lancedb.connect(DB_DIR)
    table = db.open_table(TABLE_NAME)
    client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")

    if len(sys.argv) > 1:
        ask_rag(client, embedder, table, " ".join(sys.argv[1:]))
        return

    print("[âœ“] Ready! Ask a question about Project Insight (or type 'exit'):")
    while True:
        try:
            q = input("\n> ").strip()
            if not q or q.lower() in {"exit", "quit"}:
                break
            ask_rag(client, embedder, table, q)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
