"""
Module 1 CLI Runner: Hybrid Search Plus Reranking
Demonstrates BM25 lexical search, Dense vector search, RRF & Linear Fusion,
Cross-Encoder Reranking, and LLM grounded synthesis.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
from src.hybrid_rag.hybrid_pipeline import HybridRAGPipeline
from config import settings

def run_hybrid_demo():
    print("=" * 80)
    print("🚀 ADVANCED RAG - MODULE 1: HYBRID SEARCH PLUS RERANKING")
    print("=" * 80)

    pipeline = HybridRAGPipeline()
    print(f"\n📂 Ingesting sample corpus from: {settings.SAMPLE_CORPUS_DIR}")
    num_chunks = pipeline.ingest_corpus()
    print(f"✅ Successfully indexed {num_chunks} chunks into BM25 and Dense Vector Stores.")

    test_queries = [
        "What is the smoothing constant k in Reciprocal Rank Fusion?",
        "Why is a Cross-Encoder better than a bi-encoder for passage reranking?",
        "How does PaymentEngine use Neo4j to detect fraud rings?"
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print("\n" + "-" * 80)
        print(f"🔍 QUERY {q_idx}: \"{query}\"")
        print("-" * 80)

        # 1. Compare Pure BM25 vs Pure Dense vs RRF vs Rerank
        res = pipeline.query(
            query=query,
            top_k=5,
            fusion_method="rrf",
            apply_rerank=True,
            rerank_top_n=3
        )

        print("\n📊 1. Top BM25 Lexical Candidates:")
        for c in res["bm25_candidates"][:3]:
            print(f"   [{c['chunk_id']}] Raw BM25: {c['raw_score']:.3f} | Norm: {c['normalized_score']:.3f} | Matched Terms: {c.get('matched_terms', [])}")

        print("\n📊 2. Top Dense Vector Candidates:")
        for c in res["dense_candidates"][:3]:
            print(f"   [{c['chunk_id']}] Cosine Sim: {c['raw_score']:.3f} | Norm: {c['normalized_score']:.3f}")

        print("\n📊 3. Fused Candidates (RRF k=60):")
        for c in res["fused_candidates"][:3]:
            print(f"   [{c['chunk_id']}] Fused Rank: {c['fused_rank']} | RRF Score: {c['fused_score']:.4f} (BM25 Rank: {c.get('bm25_rank')}, Dense Rank: {c.get('dense_rank')})")

        print("\n🎯 4. Final Passages After Cross-Encoder Reranking:")
        for c in res["final_passages"]:
            delta_str = f"+{c['rank_delta']}" if c['rank_delta'] > 0 else f"{c['rank_delta']}"
            print(f"   [{c['chunk_id']}] Final Rank: {c['final_rank']} | Rerank Score: {c['rerank_score']:.3f} (Rank Shift: {delta_str})")
            print(f"      Snippet: {c['content'][:140]}...")

        print("\n🤖 5. Grounded LLM Synthesis:")
        print(f"   Provider: {res['llm_provider']}")
        print(f"   Answer: {res['answer']}")
        print(f"   Citations: {[cit['chunk_id'] for cit in res['citations']]}")
        print(f"   Timings: Total={res['timings']['total_search_ms']}ms | Rerank={res['timings']['reranking_ms']}ms | Generation={res['timings']['generation_ms']}ms")

    print("\n" + "=" * 80)
    print("✅ Module 1 Execution Complete!")
    print("=" * 80)

if __name__ == "__main__":
    run_hybrid_demo()
