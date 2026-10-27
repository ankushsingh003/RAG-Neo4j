"""
Module 2 CLI Runner: GraphRAG with Neo4j
Demonstrates Knowledge Graph construction, entity linking, Cypher graph traversal,
subgraph context serialization, and hybrid Graph+Vector synthesis.
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

from src.graph_rag.graph_pipeline import GraphRAGPipeline
from config import settings

def run_graph_demo():
    print("=" * 80)
    print("🕸️ ADVANCED RAG - MODULE 2: GRAPHRAG WITH NEO4J")
    print("=" * 80)

    pipeline = GraphRAGPipeline()
    print(f"\n📂 Ingesting corpus and constructing Knowledge Graph from: {settings.SAMPLE_CORPUS_DIR}")
    stats = pipeline.ingest_corpus()
    print(f"✅ Knowledge Graph constructed:")
    print(f"   - Live Neo4j Connected: {stats['is_live_neo4j']}")
    print(f"   - Document Chunks Indexed: {stats['num_chunks_indexed']}")
    print(f"   - Unique Extracted Entities: {stats['num_unique_entities']}")
    print(f"   - Graph Relationships Created: {stats['num_relationships']}")

    test_queries = [
        "What services does AuthGateway depend on and communicate with?",
        "How does PaymentEngine use Neo4j for fraud ring detection?",
        "What architectural pattern allows Neo4j to achieve O(1) traversal complexity?"
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print("\n" + "-" * 80)
        print(f"🔍 QUERY {q_idx}: \"{query}\"")
        print("-" * 80)

        res = pipeline.query(query=query, max_hops=2, top_vector_chunks=2)

        print(f"\n🏷️ 1. Linked Query Entities: {res['anchor_entities']}")

        print(f"\n🔗 2. Discovered Knowledge Graph Triples ({len(res['graph_triples'])} relationships):")
        for triple in res["graph_triples"][:5]:
            print(f"   • ({triple['source']}) -[:{triple['relationship']}]-> ({triple['target']})")

        print(f"\n📚 3. Enriched Hybrid Contexts ({len(res['retrieved_contexts'])} items):")
        for ctx in res["retrieved_contexts"][:3]:
            print(f"   [{ctx['chunk_id']}] ({ctx.get('provenance', 'unknown')}): {ctx['content'][:120]}...")

        print("\n🤖 4. Graph-Grounded LLM Answer:")
        print(f"   Provider: {res['llm_provider']}")
        print(f"   Answer: {res['answer']}")
        print(f"   Citations: {[c['chunk_id'] for c in res['citations']]}")
        print(f"   Timings: Total={res['timings']['total_ms']}ms | Graph Traversal={res['timings']['graph_traversal_ms']}ms | Vector={res['timings']['vector_retrieval_ms']}ms")

    print("\n" + "=" * 80)
    print("✅ Module 2 Execution Complete!")
    print("=" * 80)

if __name__ == "__main__":
    run_graph_demo()
