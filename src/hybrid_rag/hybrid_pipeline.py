"""
End-to-End Hybrid RAG Pipeline
Combines BM25 Lexical Retrieval, Dense Vector Embeddings,
Rank/Score Fusion (RRF or Convex Linear), and Cross-Encoder Reranking.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.document_loader import DocumentLoader, DocumentChunk
from src.hybrid_rag.bm25_retriever import BM25Retriever
from src.hybrid_rag.dense_retriever import DenseRetriever
from src.hybrid_rag.hybrid_fusion import HybridFusion
from src.hybrid_rag.reranker import PassageReranker
from src.generator.llm_factory import LLMGenerator
from config import settings

class HybridRAGPipeline:
    def __init__(
        self,
        embedding_model: str = None,
        reranker_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.loader = DocumentLoader(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(model_name=embedding_model)
        self.fusion = HybridFusion()
        self.reranker = PassageReranker(model_name=reranker_model)
        self.generator = LLMGenerator()
        
        self.chunks: List[DocumentChunk] = []
        self.is_indexed = False

    def ingest_corpus(self, corpus_path: Optional[Path] = None) -> int:
        """
        Loads and indexes documents from a file or directory.
        Defaults to settings.SAMPLE_CORPUS_DIR if none provided.
        """
        path = corpus_path or settings.SAMPLE_CORPUS_DIR
        path = Path(path)

        if path.is_file():
            self.chunks = self.loader.load_file(path)
        elif path.is_dir():
            self.chunks = self.loader.load_directory(path)
        else:
            raise ValueError(f"Corpus path does not exist: {path}")

        if not self.chunks:
            raise ValueError(f"No text content found at: {path}")

        # Index both sparse and dense stores
        self.bm25_retriever.index_chunks(self.chunks)
        self.dense_retriever.index_chunks(self.chunks)
        self.is_indexed = True

        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        apply_rerank: bool = True,
        rerank_top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval followed by optional cross-encoder reranking.
        """
        if not self.is_indexed:
            self.ingest_corpus()

        timings = {}
        t0 = time.perf_counter()

        # Step 1: Parallel / Sequential retrieval
        t_bm25 = time.perf_counter()
        bm25_candidates = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        timings["bm25_retrieval_ms"] = round((time.perf_counter() - t_bm25) * 1000, 2)

        t_dense = time.perf_counter()
        dense_candidates = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        timings["dense_retrieval_ms"] = round((time.perf_counter() - t_dense) * 1000, 2)

        # Step 2: Fusion
        t_fusion = time.perf_counter()
        if fusion_method.lower() == "rrf":
            fused_candidates = self.fusion.reciprocal_rank_fusion(
                bm25_results=bm25_candidates,
                dense_results=dense_candidates,
                k=settings.RRF_K,
                top_n=top_k
            )
        else:
            fused_candidates = self.fusion.linear_score_fusion(
                bm25_results=bm25_candidates,
                dense_results=dense_candidates,
                alpha=alpha,
                top_n=top_k
            )
        timings["fusion_ms"] = round((time.perf_counter() - t_fusion) * 1000, 2)

        # Step 3: Cross-Encoder Reranking
        final_passages = fused_candidates
        timings["reranking_ms"] = 0.0
        if apply_rerank and fused_candidates:
            t_rerank = time.perf_counter()
            final_passages = self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                top_n=rerank_top_n
            )
            timings["reranking_ms"] = round((time.perf_counter() - t_rerank) * 1000, 2)

        timings["total_search_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "query": query,
            "bm25_candidates": bm25_candidates,
            "dense_candidates": dense_candidates,
            "fused_candidates": fused_candidates,
            "final_passages": final_passages,
            "applied_rerank": apply_rerank,
            "fusion_method": fusion_method,
            "timings": timings
        }

    def query(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        apply_rerank: bool = True,
        rerank_top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Hybrid RAG: Search + Rerank + LLM Synthesis.
        """
        search_res = self.search(
            query=query,
            top_k=top_k,
            fusion_method=fusion_method,
            alpha=alpha,
            apply_rerank=apply_rerank,
            rerank_top_n=rerank_top_n
        )

        t_gen = time.perf_counter()
        generation = self.generator.generate_answer(
            query=query,
            retrieved_contexts=search_res["final_passages"]
        )
        search_res["timings"]["generation_ms"] = round((time.perf_counter() - t_gen) * 1000, 2)
        search_res["answer"] = generation["answer"]
        search_res["citations"] = generation["citations"]
        search_res["llm_provider"] = generation["provider"]

        return search_res
