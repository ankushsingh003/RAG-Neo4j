"""
RAGAS Evaluation Engine
Evaluates RAG pipelines on four core metrics:
1. Faithfulness (Grounding of answer in retrieved context)
2. Answer Relevancy (Directness of answer to question)
3. Context Precision (Signal-to-noise ratio and ranking of relevant context)
4. Context Recall (Coverage of ground truth information)
Supports both official Ragas LLM evaluation and deterministic semantic metrics.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

class RagasEvaluator:
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self._embedding_model = embedding_model

    @property
    def model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluates a single Q&A interaction against the 4 RAGAS metrics.
        Returns normalized scores in [0.0, 1.0].
        """
        combined_context = " ".join(contexts) if contexts else ""
        
        faithfulness = self.calculate_faithfulness(answer=answer, context=combined_context)
        answer_relevancy = self.calculate_answer_relevancy(question=question, answer=answer)
        context_precision = self.calculate_context_precision(contexts=contexts, ground_truth=ground_truth)
        context_recall = self.calculate_context_recall(contexts=contexts, ground_truth=ground_truth)

        ragas_score = (faithfulness + answer_relevancy + context_precision + context_recall) / 4.0

        return {
            "faithfulness": round(float(faithfulness), 4),
            "answer_relevancy": round(float(answer_relevancy), 4),
            "context_precision": round(float(context_precision), 4),
            "context_recall": round(float(context_recall), 4),
            "ragas_score": round(float(ragas_score), 4)
        }

    def calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness = (Number of answer claims supported by context) / (Total claims).
        """
        if not answer.strip() or not context.strip():
            return 0.0

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        # Encode sentences and context
        ans_vecs = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        
        # Split context into paragraphs
        ctx_paras = [p.strip() for p in context.split("\n") if len(p.strip()) > 20]
        if not ctx_paras:
            ctx_paras = [context]
        ctx_vecs = self.model.encode(ctx_paras, convert_to_numpy=True, show_progress_bar=False)

        # Normalize
        ans_norm = ans_vecs / (np.linalg.norm(ans_vecs, axis=1, keepdims=True) + 1e-9)
        ctx_norm = ctx_vecs / (np.linalg.norm(ctx_vecs, axis=1, keepdims=True) + 1e-9)

        # Max cosine similarity per sentence to any context passage
        sim_matrix = np.dot(ans_norm, ctx_norm.T)
        max_sims = np.max(sim_matrix, axis=1)

        # Claim supported if semantic similarity >= 0.55
        supported = np.sum(max_sims >= 0.55)
        return float(supported / len(sentences))

    def calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy: Evaluates semantic alignment between prompt and response.
        """
        if not answer.strip() or not question.strip():
            return 0.0

        q_vec = self.model.encode([question], convert_to_numpy=True, show_progress_bar=False)
        a_vec = self.model.encode([answer], convert_to_numpy=True, show_progress_bar=False)

        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        a_norm = a_vec / (np.linalg.norm(a_vec) + 1e-9)

        cos_sim = float(np.dot(q_norm, a_norm.T)[0, 0])
        return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

    def calculate_context_precision(self, contexts: List[str], ground_truth: str) -> float:
        """
        Context Precision: Measures whether ground-truth relevant passages appear at top ranks.
        Evaluates Precision@k weighted by reciprocal rank.
        """
        if not contexts or not ground_truth.strip():
            return 0.0

        gt_vec = self.model.encode([ground_truth], convert_to_numpy=True, show_progress_bar=False)
        gt_norm = gt_vec / (np.linalg.norm(gt_vec) + 1e-9)

        ctx_vecs = self.model.encode(contexts, convert_to_numpy=True, show_progress_bar=False)
        ctx_norm = ctx_vecs / (np.linalg.norm(ctx_vecs, axis=1, keepdims=True) + 1e-9)

        sims = np.dot(ctx_norm, gt_norm.T).squeeze()
        if np.ndim(sims) == 0:
            sims = np.array([float(sims)])

        # High relevance threshold
        relevance_flags = [1 if s >= 0.50 else 0 for s in sims]

        if sum(relevance_flags) == 0:
            return 0.0

        score_accum = 0.0
        relevant_count = 0
        for k, flag in enumerate(relevance_flags, start=1):
            if flag == 1:
                relevant_count += 1
                precision_at_k = relevant_count / k
                score_accum += precision_at_k

        return float(score_accum / relevant_count)

    def calculate_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Context Recall: Measures what proportion of ground truth facts are captured in retrieved contexts.
        """
        if not ground_truth.strip():
            return 1.0
        if not contexts:
            return 0.0

        # Split ground truth into factual sentences
        gt_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ground_truth) if len(s.strip()) > 8]
        if not gt_sentences:
            return 1.0

        gt_vecs = self.model.encode(gt_sentences, convert_to_numpy=True, show_progress_bar=False)
        gt_norm = gt_vecs / (np.linalg.norm(gt_vecs, axis=1, keepdims=True) + 1e-9)

        ctx_vecs = self.model.encode(contexts, convert_to_numpy=True, show_progress_bar=False)
        ctx_norm = ctx_vecs / (np.linalg.norm(ctx_vecs, axis=1, keepdims=True) + 1e-9)

        sim_matrix = np.dot(gt_norm, ctx_norm.T)
        max_recall_per_fact = np.max(sim_matrix, axis=1)

        # Fact is considered recalled if maximum similarity in any retrieved chunk >= 0.50
        recalled_facts = np.sum(max_recall_per_fact >= 0.50)
        return float(recalled_facts / len(gt_sentences))
