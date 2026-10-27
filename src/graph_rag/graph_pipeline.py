"""
GraphRAG Pipeline
Integrates Neo4j Knowledge Graph traversal with Dense Vector search
for multi-hop entity reasoning and global contextual synthesis.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.document_loader import DocumentLoader, DocumentChunk
from src.graph_rag.neo4j_client import Neo4jGraphClient
from src.graph_rag.graph_builder import KnowledgeGraphBuilder
from src.graph_rag.graph_retriever import GraphRAGRetriever
from src.hybrid_rag.dense_retriever import DenseRetriever
from src.generator.llm_factory import LLMGenerator
from config import settings

class GraphRAGPipeline:
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        embedding_model: str = None
    ):
        self.loader = DocumentLoader(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        self.client = Neo4jGraphClient(uri=uri, username=username, password=password)
        self.builder = KnowledgeGraphBuilder(client=self.client)
        self.dense_retriever = DenseRetriever(model_name=embedding_model)
        self.generator = LLMGenerator()
        
        self.chunks: List[DocumentChunk] = []
        self.retriever: Optional[GraphRAGRetriever] = None
        self.is_indexed = False

    def ingest_corpus(self, corpus_path: Optional[Path] = None) -> Dict[str, Any]:
        """Loads documents, builds Neo4j Knowledge Graph, and indexes vector embeddings."""
        path = corpus_path or settings.SAMPLE_CORPUS_DIR
        path = Path(path)

        if path.is_file():
            self.chunks = self.loader.load_file(path)
        elif path.is_dir():
            self.chunks = self.loader.load_directory(path)
        else:
            raise ValueError(f"Corpus path does not exist: {path}")

        # 1. Build Knowledge Graph (Entities, Relations, Chunks)
        graph_stats = self.builder.build_from_chunks(self.chunks)

        # 2. Build Dense Vector Index for hybrid fallback
        self.dense_retriever.index_chunks(self.chunks)

        # 3. Initialize Graph Retriever
        self.retriever = GraphRAGRetriever(client=self.client, chunks=self.chunks)
        self.is_indexed = True

        return graph_stats

    def query(self, query: str, max_hops: int = 2, top_vector_chunks: int = 3) -> Dict[str, Any]:
        """
        Executes hybrid Graph + Vector retrieval and synthesizes response.
        """
        if not self.is_indexed:
            self.ingest_corpus()

        timings = {}
        t0 = time.perf_counter()

        # Step 1: Graph Traversal & Subgraph Discovery
        t_graph = time.perf_counter()
        graph_res = self.retriever.retrieve_subgraph(query=query, max_hops=max_hops)
        timings["graph_traversal_ms"] = round((time.perf_counter() - t_graph) * 1000, 2)

        # Step 2: Dense Vector Retrieval for semantic coverage
        t_dense = time.perf_counter()
        vector_res = self.dense_retriever.retrieve(query=query, top_k=top_vector_chunks)
        timings["vector_retrieval_ms"] = round((time.perf_counter() - t_dense) * 1000, 2)

        # Step 3: Combine Graph Supporting Chunks + Top Vector Chunks
        merged_chunks = []
        seen_cids = set()

        for c in graph_res["supporting_chunks"]:
            cid = c["chunk_id"]
            if cid not in seen_cids:
                seen_cids.add(cid)
                merged_chunks.append({
                    "chunk_id": cid,
                    "title": c.get("title", ""),
                    "content": c["content"],
                    "provenance": "graph_traversal"
                })

        for v in vector_res:
            cid = v["chunk_id"]
            if cid not in seen_cids:
                seen_cids.add(cid)
                merged_chunks.append({
                    "chunk_id": cid,
                    "title": v.get("title", ""),
                    "content": v["content"],
                    "provenance": "vector_similarity"
                })

        # Inject structured graph facts into context
        graph_facts_str = "\n".join(graph_res["formatted_graph_facts"])
        if graph_facts_str:
            merged_chunks.insert(0, {
                "chunk_id": "knowledge_graph_facts",
                "title": "Structured Knowledge Graph Triples",
                "content": f"The following structured entity relationships are established in the Knowledge Graph:\n{graph_facts_str}",
                "provenance": "knowledge_graph"
            })

        # Step 4: Grounded Synthesis
        t_gen = time.perf_counter()
        generation = self.generator.generate_answer(query=query, retrieved_contexts=merged_chunks)
        timings["generation_ms"] = round((time.perf_counter() - t_gen) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "query": query,
            "anchor_entities": graph_res["anchor_entities"],
            "graph_triples": graph_res["triples"],
            "formatted_graph_facts": graph_res["formatted_graph_facts"],
            "retrieved_contexts": merged_chunks,
            "answer": generation["answer"],
            "citations": generation["citations"],
            "llm_provider": generation["provider"],
            "timings": timings,
            "is_live_neo4j": self.client.is_live_neo4j
        }
