import os
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
#     "lancedb",
#     "fastembed",
#     "pandas",
#     "tantivy",
# ]
# ///

import lancedb
from fastembed import TextEmbedding
from fastmcp import FastMCP
from lancedb.rerankers import RRFReranker

DB_DIR = os.getenv("LANCEDB_DIR", "./lancedb_store")
TABLE_NAME = os.getenv("LANCEDB_TABLE", "technical_docs")
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", "./.fastembed_cache")

mcp = FastMCP("Technical Documentation RAG")

# Load once at server startup so queries return in <100ms
_embedder = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=CACHE_DIR)
_db = lancedb.connect(DB_DIR)
_table = _db.open_table(TABLE_NAME)
_reranker = RRFReranker()

# Precompute lightweight table of contents once at startup
_df = _table.to_pandas()[["path", "doc_title", "url"]].drop_duplicates(subset=["path"])
_TOPICS_INDEX = "\n".join(
    f"- `{row.path}` â€” {row.doc_title} ({row.url})"
    for row in _df.sort_values("path").itertuples()
)


@mcp.tool()
def list_project_insight_topics() -> str:
    """
    List all available indexed technical documentation pages (path, title, and URL).
    Call this first when exploring an unfamiliar topic or workflow stage so you can target
    exact page paths without making repeated exploratory searches.
    """
    return f"### Indexed Project Insight Pages ({len(_df)} pages)\n\n{_TOPICS_INDEX}"


@mcp.tool()
def get_project_insight_page(path: str) -> str:
    """
    Retrieve all indexed chunks for a specific documentation page path in document order.
    Use this after `list_project_insight_topics` when you need the complete specification
    of a page (e.g., `/workflow/step-7`, `/submit/checklist`, `/pitfalls/ava`).

    Args:
        path: Exact page path starting with `/` (e.g., `/workflow/step-2`).
    """
    rows = _table.search().where(f"path = '{path}'", prefilter=True).limit(50).to_list()
    if not rows:
        return f"No chunks found for path `{path}`. Call `list_project_insight_topics` to verify valid paths."

    # Preserve original chunk ingestion order via trailing numeric suffix in `id`
    rows.sort(key=lambda r: int(r["id"].rsplit("_", 1)[-1]) if r["id"].rsplit("_", 1)[-1].isdigit() else 0)

    header = f"# {rows[0]['doc_title']} (`{path}`)\nURL: {rows[0]['url']}\n"
    body = "\n\n---\n\n".join(f"### {r['breadcrumb']}\n{r['text']}" for r in rows)
    return f"{header}\n{body}"


@mcp.tool()
def search_project_insight_docs(query: str, top_k: int = 5) -> str:
    """
    Hybrid semantic + keyword search across indexed technical
    task-authoring documentation, worked domain examples, image captions, and video transcripts.

    Args:
        query: Specific question, CLI command, filename (`task.toml`, `instruction.md`),
               or rule (`pass@2`, `AVA`, verifier isolation, time caps) to look up.
        top_k: Number of top-ranked documentation chunks to return (default: 5).
    """
    q_vec = list(_embedder.query_embed(query))[0].tolist()
    results = (
        _table.search(query_type="hybrid")
        .vector(q_vec)
        .text(query)
        .rerank(reranker=_reranker)
        .limit(top_k)
        .to_list()
    )

    if not results:
        return "No matching documentation chunks found."

    formatted = []
    for idx, r in enumerate(results, 1):
        formatted.append(
            f"### [{idx}] {r['breadcrumb']}\n"
            f"- **Path:** `{r['path']}`\n"
            f"- **URL:** {r['url']}\n\n"
            f"{r['text']}"
        )

    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    mcp.run()
