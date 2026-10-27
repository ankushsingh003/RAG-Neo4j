"""
Comparative Benchmark Runner
Executes side-by-side RAGAS evaluation across:
1. Baseline Dense Vector RAG
2. Advanced Hybrid RAG + Cross-Encoder Reranking
3. GraphRAG with Neo4j Knowledge Graph
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.hybrid_rag.hybrid_pipeline import HybridRAGPipeline
from src.graph_rag.graph_pipeline import GraphRAGPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator
from config import settings

class BenchmarkRunner:
    def __init__(self):
        self.hybrid_pipeline = HybridRAGPipeline()
        self.graph_pipeline = GraphRAGPipeline()
        self.evaluator = RagasEvaluator()

    def load_dataset(self, dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Loads evaluation test dataset."""
        path = dataset_path or settings.EVAL_DATASET_PATH
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Runs complete comparative benchmark on the test dataset across all 3 architectures.
        """
        samples = self.load_dataset(dataset_path)

        # Ingest both pipelines
        print("Ingesting corpus into Hybrid & Graph stores...")
        self.hybrid_pipeline.ingest_corpus()
        self.graph_pipeline.ingest_corpus()

        results = {
            "baseline_vector": {"samples": [], "averages": {}},
            "hybrid_rerank": {"samples": [], "averages": {}},
            "graph_rag": {"samples": [], "averages": {}},
        }

        print(f"Evaluating {len(samples)} benchmark queries across 3 RAG architectures...\n")

        for sample in samples:
            q_id = sample["id"]
            question = sample["question"]
            ground_truth = sample["ground_truth"]

            # --- 1. Baseline Dense Vector RAG ---
            t0 = time.perf_counter()
            dense_search = self.hybrid_pipeline.dense_retriever.retrieve(question, top_k=3)
            dense_contexts = [c["content"] for c in dense_search]
            dense_gen = self.hybrid_pipeline.generator.generate_answer(question, dense_search)
            dense_time = (time.perf_counter() - t0) * 1000

            dense_metrics = self.evaluator.evaluate_sample(
                question=question,
                answer=dense_gen["answer"],
                contexts=dense_contexts,
                ground_truth=ground_truth
            )
            dense_metrics["latency_ms"] = round(dense_time, 2)
            results["baseline_vector"]["samples"].append({
                "id": q_id,
                "question": question,
                "answer": dense_gen["answer"],
                "metrics": dense_metrics
            })

            # --- 2. Advanced Hybrid RAG + Reranking ---
            t0 = time.perf_counter()
            hybrid_res = self.hybrid_pipeline.query(
                query=question,
                top_k=5,
                fusion_method="rrf",
                apply_rerank=True,
                rerank_top_n=3
            )
            hybrid_time = (time.perf_counter() - t0) * 1000
            hybrid_contexts = [c["content"] for c in hybrid_res["final_passages"]]

            hybrid_metrics = self.evaluator.evaluate_sample(
                question=question,
                answer=hybrid_res["answer"],
                contexts=hybrid_contexts,
                ground_truth=ground_truth
            )
            hybrid_metrics["latency_ms"] = round(hybrid_time, 2)
            results["hybrid_rerank"]["samples"].append({
                "id": q_id,
                "question": question,
                "answer": hybrid_res["answer"],
                "metrics": hybrid_metrics
            })

            # --- 3. GraphRAG with Neo4j ---
            t0 = time.perf_counter()
            graph_res = self.graph_pipeline.query(query=question, max_hops=2, top_vector_chunks=2)
            graph_time = (time.perf_counter() - t0) * 1000
            graph_contexts = [c["content"] for c in graph_res["retrieved_contexts"]]

            graph_metrics = self.evaluator.evaluate_sample(
                question=question,
                answer=graph_res["answer"],
                contexts=graph_contexts,
                ground_truth=ground_truth
            )
            graph_metrics["latency_ms"] = round(graph_time, 2)
            results["graph_rag"]["samples"].append({
                "id": q_id,
                "question": question,
                "answer": graph_res["answer"],
                "metrics": graph_metrics
            })

        # Calculate macro averages
        metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_score", "latency_ms"]
        for arch in ["baseline_vector", "hybrid_rerank", "graph_rag"]:
            arch_samples = results[arch]["samples"]
            n = len(arch_samples)
            avgs = {}
            for k in metric_keys:
                total_val = sum(s["metrics"][k] for s in arch_samples)
                avgs[k] = round(total_val / n, 4)
            results[arch]["averages"] = avgs

        return results
