"""
Entity and Relationship Extractor for GraphRAG
Extracts Knowledge Graph triples (Subject, Predicate, Object) and typed entity nodes
from raw text and document chunks.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ExtractedEntity:
    id: str
    name: str
    label: str
    description: str = ""

@dataclass
class ExtractedRelationship:
    source_id: str
    target_id: str
    rel_type: str
    description: str = ""
    chunk_id: str = ""

class KnowledgeExtractor:
    """
    Extracts structured entities and semantic relationships from document chunks.
    Combines high-precision regex/lexical heuristics and structured patterns.
    """
    KNOWN_ENTITY_TYPES = {
        "Reciprocal Rank Fusion": "ALGORITHM",
        "RRF": "ALGORITHM",
        "BM25": "ALGORITHM",
        "BM25 Okapi": "ALGORITHM",
        "Cross-Encoder": "MODEL_ARCHITECTURE",
        "Bi-Encoder": "MODEL_ARCHITECTURE",
        "all-MiniLM-L6-v2": "EMBEDDING_MODEL",
        "Neo4j": "DATABASE",
        "PostgreSQL": "DATABASE",
        "RedisCluster": "DATABASE",
        "ClickHouse": "DATABASE",
        "AuraDB": "DATABASE",
        "AuthGateway": "MICROSERVICE",
        "UserDirectory": "MICROSERVICE",
        "PaymentEngine": "MICROSERVICE",
        "AnalyticsOrchestrator": "MICROSERVICE",
        "KafkaBroker": "MESSAGE_BROKER",
        "Index-Free Adjacency": "ARCHITECTURE_PATTERN",
        "Cypher": "QUERY_LANGUAGE",
        "DeviceFingerprint": "SECURITY_ENTITY",
        "Step-Up Multi-Factor Authentication": "SECURITY_PROTOCOL",
        "Circuit Breaker": "RESILIENCE_PATTERN",
        "Saga Pattern": "TRANSACTION_PATTERN",
        "RAGAS": "EVALUATION_FRAMEWORK",
        "Faithfulness": "METRIC",
        "Answer Relevance": "METRIC",
        "Context Precision": "METRIC",
        "Context Recall": "METRIC"
    }

    RELATION_PATTERNS = [
        # X depends on Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+depends\s+on\s+([A-Z][a-zA-Z0-9_-]+)\b", "DEPENDS_ON"),
        # X communicates with Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+communicates\s+with\s+([A-Z][a-zA-Z0-9_-]+)\b", "COMMUNICATES_WITH"),
        # X persists records in Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+persists\s+.*?in\s+([A-Z][a-zA-Z0-9_-]+)\b", "PERSISTS_IN"),
        # X publishes events to Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+publishes\s+.*?to\s+([A-Z][a-zA-Z0-9_-]+)\b", "PUBLISHES_TO"),
        # X utilizes Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+utilizes\s+([A-Z][a-zA-Z0-9_-]+)\b", "UTILIZES"),
        # X implements Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+implements\s+([A-Z][a-zA-Z0-9_-]+)\b", "IMPLEMENTS"),
        # X achieves Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+achieves\s+([A-Z][a-zA-Z0-9_-]+)\b", "ACHIEVES"),
        # X triggers Y
        (r"\b([A-Z][a-zA-Z0-9_-]+)\s+triggers\s+([A-Z][a-zA-Z0-9_-]+)\b", "TRIGGERS"),
    ]

    def extract_from_chunk(self, chunk_id: str, content: str) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """
        Extracts entities and directed relationships found within a specific chunk text.
        """
        entities_dict: Dict[str, ExtractedEntity] = {}
        relationships: List[ExtractedRelationship] = []

        # 1. Match known domain entities
        for name, label in self.KNOWN_ENTITY_TYPES.items():
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, content, re.IGNORECASE):
                # Standardize id
                entity_id = self._canonical_id(name)
                if entity_id not in entities_dict:
                    entities_dict[entity_id] = ExtractedEntity(
                        id=entity_id,
                        name=name,
                        label=label,
                        description=f"{label} referenced in chunk {chunk_id}"
                    )

        # 2. Extract relationships via syntactic regex patterns
        for pattern_str, rel_type in self.RELATION_PATTERNS:
            for match in re.finditer(pattern_str, content, re.IGNORECASE):
                src_raw, tgt_raw = match.group(1), match.group(2)
                src_id = self._canonical_id(src_raw)
                tgt_id = self._canonical_id(tgt_raw)

                if src_id != tgt_id:
                    # Register nodes if not present
                    if src_id not in entities_dict:
                        entities_dict[src_id] = ExtractedEntity(
                            id=src_id,
                            name=src_raw,
                            label=self.KNOWN_ENTITY_TYPES.get(src_raw, "CONCEPT")
                        )
                    if tgt_id not in entities_dict:
                        entities_dict[tgt_id] = ExtractedEntity(
                            id=tgt_id,
                            name=tgt_raw,
                            label=self.KNOWN_ENTITY_TYPES.get(tgt_raw, "CONCEPT")
                        )

                    relationships.append(ExtractedRelationship(
                        source_id=src_id,
                        target_id=tgt_id,
                        rel_type=rel_type,
                        description=match.group(0),
                        chunk_id=chunk_id
                    ))

        # 3. Add explicit contextual relationships for known architectural connections
        self._add_domain_specific_edges(chunk_id, content, entities_dict, relationships)

        return list(entities_dict.values()), relationships

    def _add_domain_specific_edges(
        self,
        chunk_id: str,
        content: str,
        entities: Dict[str, ExtractedEntity],
        relationships: List[ExtractedRelationship]
    ):
        """Discovers implicit technical links documented in the enterprise texts."""
        lower_content = content.lower()

        # Hybrid Search domain
        if "reciprocal rank fusion" in lower_content or "rrf" in lower_content:
            if "bm25" in lower_content:
                relationships.append(ExtractedRelationship(
                    source_id="reciprocal_rank_fusion",
                    target_id="bm25",
                    rel_type="COMBINES_WITH",
                    description="RRF combines BM25 rankings",
                    chunk_id=chunk_id
                ))
            if "all-minilm-l6-v2" in lower_content or "dense" in lower_content:
                relationships.append(ExtractedRelationship(
                    source_id="reciprocal_rank_fusion",
                    target_id="all_minilm_l6_v2",
                    rel_type="COMBINES_WITH",
                    description="RRF combines dense embeddings",
                    chunk_id=chunk_id
                ))

        if "cross-encoder" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="cross_encoder",
                target_id="bi_encoder",
                rel_type="RERANKS_OUTPUT_OF",
                description="Cross-Encoder reranks bi-encoder candidates",
                chunk_id=chunk_id
            ))

        # Neo4j and GraphRAG domain
        if "index-free adjacency" in lower_content and "neo4j" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="neo4j",
                target_id="index_free_adjacency",
                rel_type="IMPLEMENTS",
                description="Neo4j implements Index-Free Adjacency",
                chunk_id=chunk_id
            ))

        if "cypher" in lower_content and "neo4j" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="neo4j",
                target_id="cypher",
                rel_type="USES_QUERY_LANGUAGE",
                description="Neo4j uses Cypher language",
                chunk_id=chunk_id
            ))

        # Distributed cloud domain
        if "authgateway" in lower_content and "rediscluster" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="authgateway",
                target_id="rediscluster",
                rel_type="DEPENDS_ON",
                description="AuthGateway uses RedisCluster for session caching",
                chunk_id=chunk_id
            ))
        if "authgateway" in lower_content and "userdirectory" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="authgateway",
                target_id="userdirectory",
                rel_type="COMMUNICATES_WITH",
                description="AuthGateway communicates via gRPC",
                chunk_id=chunk_id
            ))
        if "userdirectory" in lower_content and "postgresql" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="userdirectory",
                target_id="postgresql",
                rel_type="PERSISTS_IN",
                description="UserDirectory persists user records in PostgreSQL",
                chunk_id=chunk_id
            ))
        if "paymentengine" in lower_content and "neo4j" in lower_content:
            relationships.append(ExtractedRelationship(
                source_id="paymentengine",
                target_id="neo4j",
                rel_type="UTILIZES_FOR_FRAUD_DETECTION",
                description="PaymentEngine analyzes fraud graphs in Neo4j",
                chunk_id=chunk_id
            ))

    @staticmethod
    def _canonical_id(name: str) -> str:
        """Converts arbitrary entity names into clean snake_case identifiers."""
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
        return clean.strip("_")
