import lancedb
from fastembed import TextEmbedding
from lancedb.rerankers import RRFReranker
from multimodal_rag.config import CACHE_DIR, DB_DIR, EMBED_MODEL_NAME, TABLE_NAME


class HybridDocRetriever:
    """Hybrid Dense Vector (BGE) + BM25 Full-Text retriever with Reciprocal Rank Fusion."""

    def __init__(
        self,
        db_dir: str = DB_DIR,
        table_name: str = TABLE_NAME,
        embed_model: str = EMBED_MODEL_NAME,
        cache_dir: str = CACHE_DIR,
    ):
        self.embedder = TextEmbedding(model_name=embed_model, cache_dir=cache_dir)
        self.db = lancedb.connect(db_dir)
        self.table = self.db.open_table(table_name)
        self.reranker = RRFReranker()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_vec = list(self.embedder.query_embed(query))[0].tolist()
        return (
            self.table.search(query_type="hybrid")
            .vector(q_vec)
            .text(query)
            .rerank(reranker=self.reranker)
            .limit(top_k)
            .to_list()
        )
