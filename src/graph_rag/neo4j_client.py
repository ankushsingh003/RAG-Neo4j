"""
Neo4j Graph Database Client with In-Memory NetworkX Fallback
Connects to live Neo4j instance (AuraDB, Docker, Desktop) or seamlessly
falls back to an embedded in-memory graph store for zero-setup execution.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from config import settings

class Neo4jGraphClient:
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None,
        allow_fallback: bool = True
    ):
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE
        self.allow_fallback = allow_fallback

        self.driver = None
        self.is_live_neo4j = False
        self.memory_graph = nx.MultiDiGraph()
        self._init_connection()

    def _init_connection(self):
        """Attempts connection to live Neo4j; falls back to in-memory graph if unreachable."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                connection_timeout=2.0
            )
            driver.verify_connectivity()
            self.driver = driver
            self.is_live_neo4j = True
            print(f" Connected to live Neo4j instance at {self.uri}")
        except Exception as e:
            self.driver = None
            self.is_live_neo4j = False
            if self.allow_fallback:
                print(f" Live Neo4j not reachable at {self.uri} ({type(e).__name__}). Using high-performance In-Memory Graph Engine.")
            else:
                raise ConnectionError(f"Could not connect to Neo4j: {e}")

    def execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query on live Neo4j.
        """
        if not self.is_live_neo4j or not self.driver:
            raise RuntimeError("Live Neo4j instance not connected.")

        parameters = parameters or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def add_node(self, node_id: str, label: str, properties: Dict[str, Any]):
        """Creates or updates a node in the graph."""
        clean_props = {k: v for k, v in properties.items() if v is not None and k != "label"}
        if self.is_live_neo4j:
            query = f"""
            MERGE (n:`{label}` {{id: $node_id}})
            SET n += $props
            RETURN n
            """
            try:
                self.execute_cypher(query, {"node_id": node_id, "props": clean_props})
                return
            except Exception as e:
                print(f"Neo4j add_node error: {e}. Writing to fallback store.")

        # In-memory graph storage
        self.memory_graph.add_node(
            node_id,
            label=label,
            **clean_props
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Creates or updates a directed relationship between two nodes."""
        props = properties or {}
        if self.is_live_neo4j:
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            MERGE (a)-[r:`{rel_type}`]->(b)
            SET r += $props
            RETURN r
            """
            try:
                self.execute_cypher(query, {
                    "source_id": source_id,
                    "target_id": target_id,
                    "props": props
                })
                return
            except Exception as e:
                print(f"Neo4j add_relationship error: {e}. Writing to fallback store.")

        # In-memory graph storage
        self.memory_graph.add_edge(
            source_id,
            target_id,
            key=rel_type,
            rel_type=rel_type,
            **props
        )

    def get_neighbors(self, node_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves 1-hop and 2-hop connected entities and relationship types.
        """
        if self.is_live_neo4j:
            query = f"""
            MATCH path = (start {{id: $node_id}})-[r*1..{max_hops}]-(target)
            UNWIND relationships(path) AS rel
            RETURN startNode(rel).id AS source,
                   type(rel) AS relationship,
                   endNode(rel).id AS target,
                   labels(endNode(rel))[0] AS target_label,
                   endNode(rel).description AS target_description
            LIMIT 50
            """
            try:
                return self.execute_cypher(query, {"node_id": node_id})
            except Exception as e:
                print(f"Neo4j get_neighbors error: {e}. Falling back.")

        # In-memory neighborhood traversal
        results = []
        if node_id not in self.memory_graph:
            return results

        # 1-hop outgoing
        for _, target, key, data in self.memory_graph.out_edges(node_id, keys=True, data=True):
            target_data = self.memory_graph.nodes.get(target, {})
            results.append({
                "source": node_id,
                "relationship": key,
                "target": target,
                "target_label": target_data.get("label", "Entity"),
                "target_description": target_data.get("description", "")
            })

        # 1-hop incoming
        for source, _, key, data in self.memory_graph.in_edges(node_id, keys=True, data=True):
            source_data = self.memory_graph.nodes.get(source, {})
            results.append({
                "source": source,
                "relationship": key,
                "target": node_id,
                "target_label": source_data.get("label", "Entity"),
                "target_description": source_data.get("description", "")
            })

        if max_hops >= 2:
            visited_1hop = {r["target"] for r in results if r["source"] == node_id}
            for hop1 in visited_1hop:
                if hop1 == node_id:
                    continue
                for _, target2, key2, data2 in self.memory_graph.out_edges(hop1, keys=True, data=True):
                    if target2 != node_id:
                        t2_data = self.memory_graph.nodes.get(target2, {})
                        results.append({
                            "source": hop1,
                            "relationship": key2,
                            "target": target2,
                            "target_label": t2_data.get("label", "Entity"),
                            "target_description": t2_data.get("description", "")
                        })

        return results

    def get_all_graph_data(self) -> Dict[str, Any]:
        """Returns all nodes and edges for visualization in Streamlit / PyVis."""
        if self.is_live_neo4j:
            try:
                nodes_res = self.execute_cypher("MATCH (n) RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, n.description AS description LIMIT 200")
                edges_res = self.execute_cypher("MATCH (a)-[r]->(b) RETURN a.id AS source, b.id AS target, type(r) AS relationship LIMIT 300")
                return {"nodes": nodes_res, "edges": edges_res}
            except Exception:
                pass

        # In-memory export
        nodes = []
        for n, data in self.memory_graph.nodes(data=True):
            nodes.append({
                "id": str(n),
                "label": data.get("label", "Entity"),
                "name": data.get("name", str(n)),
                "description": data.get("description", "")
            })
        edges = []
        for u, v, k, data in self.memory_graph.edges(keys=True, data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "relationship": k
            })

        return {"nodes": nodes, "edges": edges}

    def clear(self):
        """Clears the graph."""
        if self.is_live_neo4j and self.driver:
            try:
                self.execute_cypher("MATCH (n) DETACH DELETE n")
            except Exception:
                pass
        self.memory_graph.clear()

    def close(self):
        """Closes live driver if active."""
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
