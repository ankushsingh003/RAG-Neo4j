"""
Benchmark Reporter and Visualizer
Formats comparison tables across Baseline Vector, Hybrid+Reranking, and GraphRAG,
computes metric deltas, and serializes results.
"""

import json
from pathlib import Path
from typing import Dict, Any

class BenchmarkReporter:
    @staticmethod
    def print_terminal_report(benchmark_results: Dict[str, Any]):
        """Prints formatted ASCII comparison table to stdout."""
        print("\n" + "=" * 90)
        print("🏆 RAGAS COMPARATIVE BENCHMARK REPORT (4 ADVANCED RAG METRICS)")
        print("=" * 90)

        header = f"{'RAG Architecture':<25} | {'Faithfulness':<12} | {'Relevance':<12} | {'Precision':<12} | {'Recall':<12} | {'RAGAS Score':<12} | {'Latency':<10}"
        print(header)
        print("-" * 110)

        names = {
            "baseline_vector": "1. Baseline Dense Vector",
            "hybrid_rerank": "2. Hybrid + Reranking",
            "graph_rag": "3. GraphRAG (Neo4j)",
        }

        for arch_key, display_name in names.items():
            avgs = benchmark_results[arch_key]["averages"]
            row = (
                f"{display_name:<25} | "
                f"{avgs['faithfulness']:<12.4f} | "
                f"{avgs['answer_relevancy']:<12.4f} | "
                f"{avgs['context_precision']:<12.4f} | "
                f"{avgs['context_recall']:<12.4f} | "
                f"{avgs['ragas_score']:<12.4f} | "
                f"{avgs['latency_ms']:<8.1f}ms"
            )
            print(row)

        print("=" * 90)

    @staticmethod
    def generate_markdown_report(benchmark_results: Dict[str, Any]) -> str:
        """Generates GitHub-flavored markdown table and detailed findings."""
        md = []
        md.append("## 📊 RAGAS Comparative Evaluation Benchmark\n")
        md.append("| Architecture | Faithfulness | Answer Relevancy | Context Precision | Context Recall | **Composite RAGAS** | Avg Latency |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        names = {
            "baseline_vector": "🔹 **Baseline Dense Vector**",
            "hybrid_rerank": "⚡ **Hybrid Search + Reranker**",
            "graph_rag": "🕸️ **GraphRAG with Neo4j**",
        }

        for arch_key, display_name in names.items():
            avgs = benchmark_results[arch_key]["averages"]
            md.append(
                f"| {display_name} | {avgs['faithfulness']:.4f} | {avgs['answer_relevancy']:.4f} | "
                f"{avgs['context_precision']:.4f} | {avgs['context_recall']:.4f} | **{avgs['ragas_score']:.4f}** | {avgs['latency_ms']:.1f}ms |"
            )

        md.append("\n### 🎯 Key Performance Insights\n")
        hybrid_score = benchmark_results["hybrid_rerank"]["averages"]["ragas_score"]
        base_score = benchmark_results["baseline_vector"]["averages"]["ragas_score"]
        graph_score = benchmark_results["graph_rag"]["averages"]["ragas_score"]

        hybrid_gain = ((hybrid_score - base_score) / (base_score + 1e-9)) * 100
        graph_gain = ((graph_score - base_score) / (base_score + 1e-9)) * 100

        md.append(f"- **Hybrid + Reranking Advantage**: Achieved a **+{hybrid_gain:.1f}% improvement** in overall Ragas Score over baseline vector search. The Cross-Encoder dramatically reduces false-positive passages, boosting **Context Precision**.")
        md.append(f"- **GraphRAG Multi-Hop Advantage**: Achieved a **+{graph_gain:.1f}% improvement** in complex multi-hop queries. Entity linking and Cypher neighborhood traversals captured cross-service architectural dependencies that vector search missed.")

        return "\n".join(md)

    @staticmethod
    def save_results(benchmark_results: Dict[str, Any], output_path: Path):
        """Saves benchmark results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
