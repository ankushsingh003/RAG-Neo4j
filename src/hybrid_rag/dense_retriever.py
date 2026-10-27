"""
Dense Vector Retriever using SentenceTransformer and Vector Similarity
Embeds document chunks, computes cosine similarity, and supports ChromaDB persistence.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.document_loader import DocumentChunk
from config import settings

class DenseRetriever:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model: Optional[SentenceTransformer] = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model to optimize startup time."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Compute embeddings for all chunks and store in normalized memory index."""
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]

        # Compute normalized embeddings for fast dot-product cosine similarity
        raw_embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # L2-normalize
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.chunk_embeddings = raw_embeddings / norms

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks by cosine similarity to query vector.
        """
        if self.chunk_embeddings is None or not self.chunks:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.chunk_embeddings, query_vec.T).squeeze()
        if np.ndim(scores) == 0:
            scores = np.array([float(scores)])

        # Rank indices descending
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            raw_score = float(scores[idx])
            # Cosine similarity in [-1, 1], map to [0, 1] for normalization
            norm_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

            results.append({
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "raw_score": raw_score,
                "normalized_score": norm_score,
                "rank": rank,
                "retriever": "dense"
            })

        return results
