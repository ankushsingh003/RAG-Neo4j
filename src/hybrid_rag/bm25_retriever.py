"""
BM25 Sparse Lexical Retriever
Tokenizes chunks and queries, computes BM25Okapi scores,
and normalizes relevance weights for hybrid fusion.
"""

import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from src.document_loader import DocumentChunk

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into lowercase alphanumeric tokens."""
        text = text.lower()
        # Preserve hyphens inside technical terms (e.g., ISO-27001, O(1), CVE-2024)
        tokens = re.findall(r"\b[a-z0-9_-]+\b", text)
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Build BM25 index over the provided document chunks."""
        self.chunks = chunks
        self.corpus_tokens = [self.tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks for the given query with raw and min-max normalized BM25 scores.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Rank by score descending
        scored_pairs = list(enumerate(scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        top_results = scored_pairs[:top_k]
        
        # Calculate min and max for normalization
        max_score = scored_pairs[0][1] if scored_pairs else 1.0
        min_score = min(scores) if len(scores) > 0 else 0.0
        denom = (max_score - min_score) if (max_score - min_score) > 0 else 1.0

        results = []
        for rank, (chunk_idx, raw_score) in enumerate(top_results, start=1):
            chunk = self.chunks[chunk_idx]
            norm_score = max(0.0, (raw_score - min_score) / denom)
            
            # Find matched query tokens
            chunk_tokens_set = set(self.corpus_tokens[chunk_idx])
            matched_terms = [t for t in set(query_tokens) if t in chunk_tokens_set]

            results.append({
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "raw_score": float(raw_score),
                "normalized_score": float(norm_score),
                "rank": rank,
                "retriever": "bm25",
                "matched_terms": matched_terms
            })

        return results
