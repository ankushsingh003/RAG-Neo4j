"""
GraphRAG Retriever
Performs entity linking on queries, multi-hop Cypher/graph traversal,
subgraph serialization, and chunk provenance retrieval.
"""

from typing import List, Dict, Any, Set, Tuple
import re
from src.graph_rag.neo4j_client import Neo4jGraphClient
from src.graph_rag.entity_extractor import KnowledgeExtractor
from src.document_loader import DocumentChunk

class GraphRAGRetriever:
    def __init__(self, client: Neo4jGraphClient, chunks: List[DocumentChunk]):
        self.client = client
        self.chunks = {c.chunk_id: c for c in chunks}
        self.extractor = KnowledgeExtractor()

    def link_entities_in_query(self, query: str) -> List[str]:
        """Identifies known entities in the query text to serve as traversal anchors."""
        matched_entity_ids = []
        for name, label in self.extractor.KNOWN_ENTITY_TYPES.items():
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, query, re.IGNORECASE):
                matched_entity_ids.append(self.extractor._canonical_id(name))

        # Check in memory graph nodes directly
        for node_id in self.client.memory_graph.nodes:
            if str(node_id).lower() in query.lower() and node_id not in matched_entity_ids:
                matched_entity_ids.append(node_id)

        return list(dict.fromkeys(matched_entity_ids))

    def retrieve_subgraph(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Extracts linked entities from query, traverses graph, and collects
        relational statements and supporting text chunks.
        """
        anchor_entities = self.link_entities_in_query(query)
        if not anchor_entities:
            # Fallback: scan for any words matching entity names
            words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
            for w in words:
                can_w = self.extractor._canonical_id(w)
                if can_w in self.client.memory_graph.nodes:
                    anchor_entities.append(can_w)

        all_triples = []
        visited_nodes = set(anchor_entities)
        supporting_chunk_ids: Set[str] = set()

        for entity_id in anchor_entities:
            neighbors = self.client.get_neighbors(entity_id, max_hops=max_hops)
            for item in neighbors:
                src = item["source"]
                tgt = item["target"]
                rel = item["relationship"]
                
                # If target or source is a Chunk node, record provenance
                if str(src).endswith(tuple(f"_chunk_{i:02d}" for i in range(100))):
                    supporting_chunk_ids.add(str(src))
                    continue
                if str(tgt).endswith(tuple(f"_chunk_{i:02d}" for i in range(100))):
                    supporting_chunk_ids.add(str(tgt))
                    continue

                visited_nodes.add(src)
                visited_nodes.add(tgt)
                all_triples.append({
                    "source": src,
                    "relationship": rel,
                    "target": tgt,
                    "target_label": item.get("target_label", "Entity"),
                    "description": item.get("target_description", "")
                })

        # Deduplicate triples
        unique_triples = []
        seen_keys = set()
        for t in all_triples:
            key = (t["source"], t["relationship"], t["target"])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_triples.append(t)

        # Retrieve text chunks linked to visited entities
        for nid in visited_nodes:
            # Check incoming MENTIONS edges from Chunks
            if not self.client.is_live_neo4j:
                for src_node, _, key in self.client.memory_graph.in_edges(nid, keys=True):
                    if key == "MENTIONS" and src_node in self.chunks:
                        supporting_chunk_ids.add(src_node)

        supporting_chunks = [self.chunks[cid] for cid in supporting_chunk_ids if cid in self.chunks]

        # Format graph paths as text facts
        formatted_facts = []
        for t in unique_triples:
            formatted_facts.append(f"({t['source']}) -[:{t['relationship']}]-> ({t['target']})")

        return {
            "query": query,
            "anchor_entities": anchor_entities,
            "triples": unique_triples,
            "formatted_graph_facts": formatted_facts,
            "supporting_chunks": [c.to_dict() for c in supporting_chunks]
        }
