"""
Hybrid Candidate Fusion Algorithms:
1. Reciprocal Rank Fusion (RRF)
2. Convex Linear Combination (Weighted Score Fusion)
"""

from typing import List, Dict, Any, Literal

class HybridFusion:
    @staticmethod
    def reciprocal_rank_fusion(
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        k: int = 60,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merges BM25 and Dense candidates using Reciprocal Rank Fusion (RRF).
        Formula: RRF(d) = sum_{retriever} [ 1.0 / (k + rank) ]
        """
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Process BM25 rankings
        for item in bm25_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            score_contrib = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "title": item["title"],
                    "content": item["content"],
                    "metadata": item.get("metadata", {}),
                    "bm25_rank": rank,
                    "bm25_score": item["raw_score"],
                    "bm25_norm": item["normalized_score"],
                    "dense_rank": None,
                    "dense_score": None,
                    "dense_norm": None,
                }
            else:
                chunk_map[cid]["bm25_rank"] = rank
                chunk_map[cid]["bm25_score"] = item["raw_score"]
                chunk_map[cid]["bm25_norm"] = item["normalized_score"]

        # Process Dense rankings
        for item in dense_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            score_contrib = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "title": item["title"],
                    "content": item["content"],
                    "metadata": item.get("metadata", {}),
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_norm": None,
                    "dense_rank": rank,
                    "dense_score": item["raw_score"],
                    "dense_norm": item["normalized_score"],
                }
            else:
                chunk_map[cid]["dense_rank"] = rank
                chunk_map[cid]["dense_score"] = item["raw_score"]
                chunk_map[cid]["dense_norm"] = item["normalized_score"]

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results = []
        for rank, cid in enumerate(sorted_cids[:top_n], start=1):
            info = chunk_map[cid]
            info["fused_score"] = rrf_scores[cid]
            info["fused_rank"] = rank
            info["fusion_method"] = "rrf"
            fused_results.append(info)

        return fused_results

    @staticmethod
    def linear_score_fusion(
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        alpha: float = 0.5,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Convex linear combination of normalized scores:
        Score(d) = alpha * Dense_norm + (1 - alpha) * BM25_norm
        """
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Ingest BM25 normalized scores
        for item in bm25_results:
            cid = item["chunk_id"]
            chunk_map[cid] = {
                "chunk_id": cid,
                "title": item["title"],
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "bm25_rank": item["rank"],
                "bm25_score": item["raw_score"],
                "bm25_norm": item["normalized_score"],
                "dense_rank": None,
                "dense_score": None,
                "dense_norm": 0.0,
            }

        # Ingest Dense normalized scores
        for item in dense_results:
            cid = item["chunk_id"]
            if cid in chunk_map:
                chunk_map[cid]["dense_rank"] = item["rank"]
                chunk_map[cid]["dense_score"] = item["raw_score"]
                chunk_map[cid]["dense_norm"] = item["normalized_score"]
            else:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "title": item["title"],
                    "content": item["content"],
                    "metadata": item.get("metadata", {}),
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_norm": 0.0,
                    "dense_rank": item["rank"],
                    "dense_score": item["raw_score"],
                    "dense_norm": item["normalized_score"],
                }

        # Compute fused score
        for cid, info in chunk_map.items():
            bm25_n = info["bm25_norm"] or 0.0
            dense_n = info["dense_norm"] or 0.0
            info["fused_score"] = (alpha * dense_n) + ((1.0 - alpha) * bm25_n)

        # Sort descending by fused score
        sorted_items = sorted(chunk_map.values(), key=lambda x: x["fused_score"], reverse=True)

        fused_results = []
        for rank, item in enumerate(sorted_items[:top_n], start=1):
            item["fused_rank"] = rank
            item["fusion_method"] = f"linear_alpha_{alpha:.2f}"
            fused_results.append(item)

        return fused_results
