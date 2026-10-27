"""
Module 3 CLI Runner: Evaluate with RAGAS
Runs standardized comparative benchmark across:
1. Baseline Dense Vector RAG
2. Hybrid Search + Cross-Encoder Reranking
3. GraphRAG with Neo4j
Evaluates Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
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

from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.reporter import BenchmarkReporter
from config import settings

def run_evaluation_demo():
    print("=" * 90)
    print("📊 ADVANCED RAG - MODULE 3: EVALUATE WITH RAGAS")
    print("=" * 90)

    runner = BenchmarkRunner()
    print(f"\n📂 Loading evaluation dataset from: {settings.EVAL_DATASET_PATH}")
    results = runner.run_benchmark()

    # 1. Print formatted ASCII terminal report
    BenchmarkReporter.print_terminal_report(results)

    # 2. Generate Markdown report and save to disk
    report_md = BenchmarkReporter.generate_markdown_report(results)
    output_json = settings.DATA_DIR / "benchmark_results.json"
    BenchmarkReporter.save_results(results, output_json)
    print(f"\n💾 Benchmark metrics successfully saved to: {output_json}")

    print("\n" + "=" * 90)
    print("📋 SUMMARY REPORT:")
    print("=" * 90)
    print(report_md)
    print("\n" + "=" * 90)
    print("✅ Module 3 Execution Complete!")
    print("=" * 90)

if __name__ == "__main__":
    run_evaluation_demo()
