import os
from dotenv import load_dotenv
load_dotenv()
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lancedb",
#     "fastembed",
#     "openai",
#     "ragas>=0.2.10",
#     "langchain-openai<0.4",
#     "langchain-community<0.4",
#     "tantivy",
# ]
# ///

import json
from collections import defaultdict
from pathlib import Path
from typing import List

import lancedb
from fastembed import TextEmbedding
from lancedb.rerankers import RRFReranker
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextRecall, Faithfulness

DB_DIR = "lancedb_store"
TABLE_NAME = os.getenv("LANCEDB_TABLE", "technical_docs")
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", "./.fastembed_cache")
EVAL_DATASET_PATH = Path("eval_golden_set.json")
REPORT_PATH = Path("eval_report.json")

LLAMA_SERVER_URL = "http://localhost:8080/v1"

SYSTEM_PROMPT = """You are a precise technical assistant for technical documentation and benchmark authoring.
Answer using ONLY the provided reference context. Cite the exact source path for every claim."""


class LocalFastEmbedWrapper(Embeddings):
    """Lightweight LangChain Embeddings adapter around fastembed without langchain-community."""

    def __init__(self, embedder: TextEmbedding):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.embedder.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.embedder.query_embed(text))[0].tolist()


def retrieve_and_generate(client: OpenAI, embedder: TextEmbedding, table, question: str, top_k: int = 5):
    q_vec = list(embedder.query_embed(question))[0].tolist()
    reranker = RRFReranker()
    chunks = (
        table.search(query_type="hybrid")
        .vector(q_vec)
        .text(question)
        .rerank(reranker=reranker)
        .limit(top_k)
        .to_list()
    )

    contexts = [c["text"] for c in chunks]
    paths = [c["path"] for c in chunks]

    formatted_context = "\n\n---\n\n".join(
        f"[{i}] {c['breadcrumb']} (Source: {c['path']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )

    res = client.chat.completions.create(
        model="gemma-4",
        temperature=0.1,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Reference Context:\n\n{formatted_context}\n\nQuestion: {question}"},
        ],
    )
    answer = res.choices[0].message.content.strip()
    return answer, contexts, paths


def main():
    golden = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))

    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=CACHE_DIR)
    db = lancedb.connect(DB_DIR)
    table = db.open_table(TABLE_NAME)
    raw_client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")

    samples = []
    retrieval_stats = defaultdict(lambda: {"hits": 0, "mrr": 0.0, "total": 0})
    detailed_rows = []

    print(f"[*] Running RAG pipeline on {len(golden)} evaluation questions...\n", flush=True)

    for idx, item in enumerate(golden, 1):
        q = item["question"]
        expected_path = item["expected_path"]
        modality = item["modality"]

        answer, contexts, retrieved_paths = retrieve_and_generate(raw_client, embedder, table, q, top_k=5)

        valid_paths = item.get("valid_paths", [expected_path])
        matching_ranks = [
            retrieved_paths.index(p) + 1 for p in valid_paths if p in retrieved_paths
        ]
        hit = len(matching_ranks) > 0
        rr = 1.0 / min(matching_ranks) if hit else 0.0

        retrieval_stats[modality]["total"] += 1
        retrieval_stats[modality]["hits"] += int(hit)
        retrieval_stats[modality]["mrr"] += rr

        retrieval_stats["ALL"]["total"] += 1
        retrieval_stats["ALL"]["hits"] += int(hit)
        retrieval_stats["ALL"]["mrr"] += rr

        samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=item["reference_answer"],
            )
        )
        detailed_rows.append(
            {
                "id": item["id"],
                "modality": modality,
                "question": q,
                "expected_path": expected_path,
                "retrieved_paths": retrieved_paths,
                "hit_at_5": hit,
                "mrr": round(rr, 3),
                "generated_answer": answer,
            }
        )
        status = f"HIT (rank {min(matching_ranks)})" if hit else "MISS"
        print(f"  [{idx:02d}/{len(golden)}] [{status}] ({modality}) {q[:65]}...")

    print("\n=== Deterministic Retrieval Summary (Top-5 Hybrid + RRF) ===")
    for mod, s in retrieval_stats.items():
        recall = s["hits"] / s["total"] if s["total"] else 0
        mrr = s["mrr"] / s["total"] if s["total"] else 0
        print(f"  {mod:<18} | Recall@5: {recall:.1%} ({s['hits']}/{s['total']}) | MRR: {mrr:.3f}")

    print("\n[*] Scoring Faithfulness, Answer Relevancy, and Context Recall with Ragas...", flush=True)
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gemma-4",
            base_url=LLAMA_SERVER_URL,
            api_key="not-needed",
            temperature=0,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(LocalFastEmbedWrapper(embedder))

    dataset = EvaluationDataset(samples=samples)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print("\n=== Ragas Generation & Context Scores ===")
    print(ragas_result)

    ragas_df = ragas_result.to_pandas()
    for i, row in enumerate(detailed_rows):
        row["faithfulness"] = float(ragas_df.loc[i, "faithfulness"])
        row["answer_relevancy"] = float(ragas_df.loc[i, "answer_relevancy"])
        row["context_recall"] = float(ragas_df.loc[i, "context_recall"])

    REPORT_PATH.write_text(json.dumps(detailed_rows, indent=2), encoding="utf-8")
    print(f"\n[âœ“] Full per-question diagnostic report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

