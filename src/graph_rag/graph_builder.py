"""
Knowledge Graph Builder
Orchestrates entity extraction, node/edge generation, and ingestion
into the Neo4j database or fallback memory graph.
"""

from typing import List, Dict, Any, Tuple
from src.document_loader import DocumentChunk
from src.graph_rag.neo4j_client import Neo4jGraphClient
from src.graph_rag.entity_extractor import KnowledgeExtractor, ExtractedEntity, ExtractedRelationship

class KnowledgeGraphBuilder:
    def __init__(self, client: Neo4jGraphClient):
        self.client = client
        self.extractor = KnowledgeExtractor()
        self.entity_catalog: Dict[str, ExtractedEntity] = {}
        self.chunk_entity_map: Dict[str, List[str]] = {}

    def build_from_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Parses all chunks, extracts nodes and relationships, and ingests them into the graph.
        """
        self.client.clear()
        total_relationships = 0
        all_entities = set()

        for chunk in chunks:
            # 1. Add Chunk node
            self.client.add_node(
                node_id=chunk.chunk_id,
                label="Chunk",
                properties={
                    "name": chunk.chunk_id,
                    "title": chunk.title,
                    "doc_id": chunk.doc_id,
                    "content": chunk.content[:200]
                }
            )

            # 2. Extract Entities and Triples
            entities, rels = self.extractor.extract_from_chunk(chunk.chunk_id, chunk.content)
            self.chunk_entity_map[chunk.chunk_id] = [e.id for e in entities]

            # 3. Add Entity nodes
            for ent in entities:
                self.entity_catalog[ent.id] = ent
                all_entities.add(ent.id)
                self.client.add_node(
                    node_id=ent.id,
                    label=ent.label,
                    properties={
                        "name": ent.name,
                        "label": ent.label,
                        "description": ent.description
                    }
                )

                # Link Chunk to Entity
                self.client.add_relationship(
                    source_id=chunk.chunk_id,
                    target_id=ent.id,
                    rel_type="MENTIONS",
                    properties={"chunk_id": chunk.chunk_id}
                )
                total_relationships += 1

            # 4. Add Extracted Entity-to-Entity Relationships
            for rel in rels:
                self.client.add_relationship(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    rel_type=rel.rel_type,
                    properties={
                        "description": rel.description,
                        "source_chunk": rel.chunk_id
                    }
                )
                total_relationships += 1

        summary = {
            "num_chunks_indexed": len(chunks),
            "num_unique_entities": len(all_entities),
            "num_relationships": total_relationships,
            "is_live_neo4j": self.client.is_live_neo4j
        }
        return summary
