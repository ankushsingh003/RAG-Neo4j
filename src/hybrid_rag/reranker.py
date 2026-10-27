"""
Cross-Encoder Passage Reranker
Computes token-to-token cross-attention between query and retrieved candidates,
re-ordering them with deep semantic precision.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from config import settings

class PassageReranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load cross-encoder to conserve memory during startup."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate chunks using the Cross-Encoder.
        """
        if not candidates:
            return []

        # Form [query, passage] pairs
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # In case single candidate
        if np.ndim(scores) == 0:
            scores = np.array([float(scores)])

        # Attach rerank score and original rank
        scored_candidates = []
        for i, cand in enumerate(candidates):
            item = dict(cand)
            item["rerank_score"] = float(scores[i])
            item["pre_rerank_rank"] = cand.get("fused_rank", i + 1)
            scored_candidates.append(item)

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        reranked_results = []
        for new_rank, item in enumerate(scored_candidates[:top_n], start=1):
            item["final_rank"] = new_rank
            item["rank_delta"] = item["pre_rerank_rank"] - new_rank
            reranked_results.append(item)

        return reranked_results
