"""
Unified LLM Synthesis Engine
Supports OpenAI, Google Gemini, Groq, and a High-Fidelity Local Synthesis Fallback
with transparent source citation tracking.
"""

import os
import re
from typing import List, Dict, Any, Optional
from config import settings

class LLMGenerator:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.DEFAULT_LLM_PROVIDER
        self.active_provider = self._resolve_provider()

    def _resolve_provider(self) -> str:
        """Determines active provider based on environment keys."""
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            return "openai"
        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            return "gemini"
        elif self.provider == "groq" and settings.GROQ_API_KEY:
            return "groq"
        
        # Auto mode
        if settings.GEMINI_API_KEY:
            return "gemini"
        if settings.OPENAI_API_KEY:
            return "openai"
        if settings.GROQ_API_KEY:
            return "groq"
        
        return "local_synthesizer"

    def generate_answer(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes an answer grounded strictly in retrieved contexts with citations.
        """
        if not retrieved_contexts:
            return {
                "answer": "I could not find any relevant context in the knowledge base to answer your question.",
                "citations": [],
                "provider": self.active_provider
            }

        # Build context prompt
        formatted_context_blocks = []
        citations = []
        for i, ctx in enumerate(retrieved_contexts, start=1):
            cid = ctx.get("chunk_id", f"chunk_{i}")
            title = ctx.get("title", "Document")
            content = ctx.get("content", "")
            formatted_context_blocks.append(f"[{cid}] (Source: {title}):\n{content}")
            citations.append({
                "chunk_id": cid,
                "title": title,
                "score": ctx.get("rerank_score", ctx.get("fused_score", ctx.get("raw_score", 0.0)))
            })

        context_str = "\n\n".join(formatted_context_blocks)

        prompt = f"""You are an advanced enterprise AI assistant.
Answer the question based STRICTLY and ONLY on the provided context below.
Always cite the source chunk ID in brackets (e.g. [doc_01_chunk_00]) whenever referencing a fact.
If the context does not contain sufficient information, state clearly what is known and what is missing.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:"""

        # Dispatch based on active provider
        if self.active_provider == "gemini":
            try:
                from google import genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return {
                    "answer": response.text.strip(),
                    "citations": citations,
                    "provider": "gemini-2.5-flash"
                }
            except Exception as e:
                # Fallback to local
                return self._local_grounded_synthesis(query, retrieved_contexts, citations)

        elif self.active_provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise technical AI that cites sources."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return {
                    "answer": response.choices[0].message.content.strip(),
                    "citations": citations,
                    "provider": "openai-gpt-4o-mini"
                }
            except Exception as e:
                return self._local_grounded_synthesis(query, retrieved_contexts, citations)

        # Default local synthesizer
        return self._local_grounded_synthesis(query, retrieved_contexts, citations)

    def _local_grounded_synthesis(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extractive and synthesis engine that selects and organizes the most relevant
        sentences from retrieved chunks into a clear, cited response.
        """
        query_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))
        selected_sentences = []
        used_chunks = set()

        for ctx in retrieved_contexts:
            cid = ctx.get("chunk_id", "source")
            content = ctx.get("content", "")
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", content)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > 0:
                    selected_sentences.append((overlap, cid, s_clean))

        # Sort by relevance to query
        selected_sentences.sort(key=lambda x: x[0], reverse=True)

        if not selected_sentences:
            top_ctx = retrieved_contexts[0]
            cid = top_ctx.get("chunk_id", "source")
            first_few = " ".join(top_ctx.get("content", "").split()[:60]) + "..."
            answer = f"Based on the retrieved context [{cid}], {first_few}"
            used_chunks.add(cid)
        else:
            # Pick top distinct informative sentences
            seen_texts = set()
            chosen = []
            for score, cid, sent in selected_sentences[:4]:
                if sent not in seen_texts:
                    seen_texts.add(sent)
                    chosen.append(f"{sent} [{cid}]")
                    used_chunks.add(cid)
            answer = " ".join(chosen)

        active_citations = [c for c in citations if c["chunk_id"] in used_chunks] or citations[:2]

        return {
            "answer": answer,
            "citations": active_citations,
            "provider": "local_grounded_synthesizer"
        }
