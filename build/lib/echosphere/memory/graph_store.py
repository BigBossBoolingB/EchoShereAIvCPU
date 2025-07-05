"""
Neo4j hybrid vector-graph database integration.

This module provides the GraphStore class for managing both symbolic relationships
and VSA hypervector properties in Neo4j, enabling hybrid queries that combine
graph traversal and vector similarity search.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
import torch
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from .vsa import VSAMemory, HyperVector


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""

    id: int
    labels: List[str]
    properties: Dict[str, Any]
    hypervector: Optional[torch.Tensor] = None


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""

    id: int
    type: str
    start_node_id: int
    end_node_id: int
    properties: Dict[str, Any]


class GraphStore:
    """
    Neo4j-based hybrid vector-graph database.

    Combines symbolic graph relationships with VSA hypervector properties
    to enable both logical traversal and semantic similarity search.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        vsa_memory: Optional[VSAMemory] = None,
    ):
        """
        Initialize the GraphStore.

        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
            database: Database name
            vsa_memory: VSA memory system for hypervector operations
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self.vsa_memory = vsa_memory or VSAMemory()
        self.logger = logging.getLogger("echosphere.graph_store")

        self.is_connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3

        # Initialize mock mode attributes
        self.mock_nodes: Dict[int, GraphNode] = {}
        self.mock_relationships: Dict[int, GraphRelationship] = {}
        self.mock_node_counter = 1
        self.mock_rel_counter = 1

    def connect(self) -> bool:
        """
        Connect to the Neo4j database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )

            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.is_connected = True
                    self.logger.info("Successfully connected to Neo4j")
                    self._initialize_schema()
                    return True

        except (ServiceUnavailable, AuthError) as e:
            self.connection_attempts += 1
            self.logger.warning(
                f"Failed to connect to Neo4j (attempt {self.connection_attempts}): {e}"
            )

            if self.connection_attempts >= self.max_connection_attempts:
                self.logger.error("Max connection attempts reached. Using mock mode.")
                self._enable_mock_mode()
                return False

        return False

    def _enable_mock_mode(self) -> None:
        """Enable mock mode when Neo4j is not available."""
        self.is_connected = False
        self.mock_nodes: Dict[int, GraphNode] = {}
        self.mock_relationships: Dict[int, GraphRelationship] = {}
        self.mock_node_counter = 1
        self.mock_rel_counter = 1
        self.logger.info("GraphStore running in mock mode (Neo4j not available)")

    def _initialize_schema(self) -> None:
        """Initialize the Neo4j schema with constraints and indexes."""
        if not self.is_connected:
            return

        schema_queries = [
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX concept_hypervector IF NOT EXISTS FOR (c:Concept) ON (c.hypervector_dim)",
            "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r]-() ON (r.type)",
            """
            CREATE VECTOR INDEX concept_vectors IF NOT EXISTS
            FOR (c:Concept) ON (c.hypervector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 10000,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
        ]

        with self.driver.session(database=self.database) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                except Exception as e:
                    self.logger.debug(f"Schema query failed (may be expected): {e}")

    def disconnect(self) -> None:
        """Disconnect from the Neo4j database."""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            self.logger.info("Disconnected from Neo4j")

    def create_concept_node(
        self, concept: str, properties: Optional[Dict[str, Any]] = None
    ) -> GraphNode:
        """
        Create a concept node with associated hypervector.

        Args:
            concept: Concept name
            properties: Additional node properties

        Returns:
            Created GraphNode
        """
        hypervector = self.vsa_memory.encode_concept(concept)

        node_properties = properties or {}
        node_properties.update(
            {
                "name": concept,
                "created_at": time.time(),
                "hypervector_dim": self.vsa_memory.dimensions,
            }
        )

        hypervector_list = hypervector.vector.cpu().numpy().tolist()
        node_properties["hypervector"] = hypervector_list

        if self.is_connected:
            return self._create_concept_node_neo4j(concept, node_properties)
        else:
            return self._create_concept_node_mock(concept, node_properties)

    def _create_concept_node_neo4j(
        self, concept: str, properties: Dict[str, Any]
    ) -> GraphNode:
        """Create concept node in Neo4j."""
        query = """
        MERGE (c:Concept {name: $name})
        SET c += $properties
        RETURN id(c) as node_id, labels(c) as labels, properties(c) as props
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=concept, properties=properties)
            record = result.single()

            if record:
                return GraphNode(
                    id=record["node_id"],
                    labels=record["labels"],
                    properties=record["props"],
                    hypervector=torch.tensor(properties["hypervector"]),
                )
            else:
                raise RuntimeError(f"Failed to create concept node: {concept}")

    def _create_concept_node_mock(
        self, concept: str, properties: Dict[str, Any]
    ) -> GraphNode:
        """Create concept node in mock mode."""
        node_id = self.mock_node_counter
        self.mock_node_counter += 1

        node = GraphNode(
            id=node_id,
            labels=["Concept"],
            properties=properties,
            hypervector=torch.tensor(properties["hypervector"]),
        )

        self.mock_nodes[node_id] = node
        return node

    def create_relationship(
        self,
        subject: str,
        relation: str,
        object_: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphRelationship:
        """
        Create a relationship between two concepts.

        Args:
            subject: Subject concept name
            relation: Relationship type
            object_: Object concept name
            properties: Additional relationship properties

        Returns:
            Created GraphRelationship
        """
        subject_node = self.get_or_create_concept(subject)
        object_node = self.get_or_create_concept(object_)

        rel_properties = properties or {}
        rel_properties.update({"created_at": time.time(), "relation_type": relation})

        if self.is_connected:
            return self._create_relationship_neo4j(
                subject_node.id, object_node.id, relation, rel_properties
            )
        else:
            return self._create_relationship_mock(
                subject_node.id, object_node.id, relation, rel_properties
            )

    def _create_relationship_neo4j(
        self, start_id: int, end_id: int, relation: str, properties: Dict[str, Any]
    ) -> GraphRelationship:
        """Create relationship in Neo4j."""
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $start_id AND id(b) = $end_id
        CREATE (a)-[r:{relation}]->(b)
        SET r += $properties
        RETURN id(r) as rel_id, type(r) as rel_type, properties(r) as props
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query, start_id=start_id, end_id=end_id, properties=properties
            )
            record = result.single()

            if record:
                return GraphRelationship(
                    id=record["rel_id"],
                    type=record["rel_type"],
                    start_node_id=start_id,
                    end_node_id=end_id,
                    properties=record["props"],
                )
            else:
                raise RuntimeError(f"Failed to create relationship: {relation}")

    def _create_relationship_mock(
        self, start_id: int, end_id: int, relation: str, properties: Dict[str, Any]
    ) -> GraphRelationship:
        """Create relationship in mock mode."""
        rel_id = self.mock_rel_counter
        self.mock_rel_counter += 1

        relationship = GraphRelationship(
            id=rel_id,
            type=relation,
            start_node_id=start_id,
            end_node_id=end_id,
            properties=properties,
        )

        self.mock_relationships[rel_id] = relationship
        return relationship

    def get_or_create_concept(self, concept: str) -> GraphNode:
        """Get existing concept node or create new one."""
        existing = self.find_concept(concept)
        if existing:
            return existing
        else:
            return self.create_concept_node(concept)

    def find_concept(self, concept: str) -> Optional[GraphNode]:
        """Find a concept node by name."""
        if self.is_connected:
            return self._find_concept_neo4j(concept)
        else:
            return self._find_concept_mock(concept)

    def _find_concept_neo4j(self, concept: str) -> Optional[GraphNode]:
        """Find concept in Neo4j."""
        query = """
        MATCH (c:Concept {name: $name})
        RETURN id(c) as node_id, labels(c) as labels, properties(c) as props
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=concept)
            record = result.single()

            if record:
                props = record["props"]
                hypervector = (
                    torch.tensor(props.get("hypervector", []))
                    if props.get("hypervector")
                    else None
                )

                return GraphNode(
                    id=record["node_id"],
                    labels=record["labels"],
                    properties=props,
                    hypervector=hypervector,
                )

        return None

    def _find_concept_mock(self, concept: str) -> Optional[GraphNode]:
        """Find concept in mock mode."""
        for node in self.mock_nodes.values():
            if node.properties.get("name") == concept:
                return node
        return None

    def find_similar_concepts(
        self, concept: str, threshold: float = 0.7, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find concepts similar to the given concept using hypervector similarity.

        Args:
            concept: Query concept name
            threshold: Similarity threshold
            limit: Maximum number of results

        Returns:
            List of (concept_name, similarity_score) tuples
        """
        query_hypervector = self.vsa_memory.encode_concept(concept)

        if self.is_connected:
            return self._find_similar_concepts_neo4j(
                query_hypervector, threshold, limit
            )
        else:
            return self._find_similar_concepts_mock(query_hypervector, threshold, limit)

    def _find_similar_concepts_neo4j(
        self, query_hypervector: HyperVector, threshold: float, limit: int
    ) -> List[Tuple[str, float]]:
        """Find similar concepts in Neo4j using vector similarity."""
        query = """
        MATCH (c:Concept)
        WHERE c.hypervector IS NOT NULL
        RETURN c.name as name, c.hypervector as hypervector
        """

        similar_concepts = []
        query_tensor = query_hypervector.vector

        with self.driver.session(database=self.database) as session:
            result = session.run(query)

            for record in result:
                concept_name = record["name"]
                hypervector_list = record["hypervector"]

                if hypervector_list:
                    concept_tensor = torch.tensor(hypervector_list)
                    similarity = self.vsa_memory.similarity(
                        query_tensor, concept_tensor
                    )

                    if similarity >= threshold:
                        similar_concepts.append((concept_name, similarity))

        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        return similar_concepts[:limit]

    def _find_similar_concepts_mock(
        self, query_hypervector: HyperVector, threshold: float, limit: int
    ) -> List[Tuple[str, float]]:
        """Find similar concepts in mock mode."""
        similar_concepts = []
        query_tensor = query_hypervector.vector

        for node in self.mock_nodes.values():
            if node.hypervector is not None:
                concept_name = node.properties.get("name")
                similarity = self.vsa_memory.similarity(query_tensor, node.hypervector)

                if similarity >= threshold and concept_name:
                    similar_concepts.append((concept_name, similarity))

        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        return similar_concepts[:limit]

    def find_relationships(
        self, concept: str, relation_type: Optional[str] = None, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Find relationships for a concept.

        Args:
            concept: Concept name
            relation_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationship dictionaries
        """
        if self.is_connected:
            return self._find_relationships_neo4j(concept, relation_type, direction)
        else:
            return self._find_relationships_mock(concept, relation_type, direction)

    def _find_relationships_neo4j(
        self, concept: str, relation_type: Optional[str], direction: str
    ) -> List[Dict[str, Any]]:
        """Find relationships in Neo4j."""
        if direction == "outgoing":
            pattern = "(c:Concept {name: $concept})-[r]->(target)"
        elif direction == "incoming":
            pattern = "(source)-[r]->(c:Concept {name: $concept})"
        else:  # both
            pattern = "(c:Concept {name: $concept})-[r]-(other)"

        query = f"MATCH {pattern}"

        if relation_type:
            query += f" WHERE type(r) = $relation_type"

        if direction == "both":
            query += " RETURN type(r) as relation, other.name as other_concept, startNode(r).name as subject, endNode(r).name as object"
        elif direction == "outgoing":
            query += " RETURN type(r) as relation, target.name as object"
        else:  # incoming
            query += " RETURN type(r) as relation, source.name as subject"

        params = {"concept": concept}
        if relation_type:
            params["relation_type"] = relation_type

        relationships = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)

            for record in result:
                rel_dict = dict(record)
                relationships.append(rel_dict)

        return relationships

    def _find_relationships_mock(
        self, concept: str, relation_type: Optional[str], direction: str
    ) -> List[Dict[str, Any]]:
        """Find relationships in mock mode."""
        concept_node = self._find_concept_mock(concept)
        if not concept_node:
            return []

        relationships = []

        for rel in self.mock_relationships.values():
            include_rel = False
            rel_dict = {}

            if (
                direction in ["outgoing", "both"]
                and rel.start_node_id == concept_node.id
            ):
                if not relation_type or rel.type == relation_type:
                    target_node = self.mock_nodes.get(rel.end_node_id)
                    if target_node:
                        rel_dict = {
                            "relation": rel.type,
                            "object": target_node.properties.get("name"),
                            "subject": concept,
                        }
                        include_rel = True

            if direction in ["incoming", "both"] and rel.end_node_id == concept_node.id:
                if not relation_type or rel.type == relation_type:
                    source_node = self.mock_nodes.get(rel.start_node_id)
                    if source_node:
                        rel_dict = {
                            "relation": rel.type,
                            "subject": source_node.properties.get("name"),
                            "object": concept,
                        }
                        include_rel = True

            if include_rel:
                relationships.append(rel_dict)

        return relationships

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.is_connected:
            return self._get_stats_neo4j()
        else:
            return self._get_stats_mock()

    def _get_stats_neo4j(self) -> Dict[str, Any]:
        """Get Neo4j statistics."""
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "concept_nodes": "MATCH (c:Concept) RETURN count(c) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
        }

        stats = {"connected": True, "mode": "neo4j"}

        with self.driver.session(database=self.database) as session:
            for stat_name, query in queries.items():
                try:
                    result = session.run(query)
                    record = result.single()
                    stats[stat_name] = record["count"] if record else 0
                except Exception as e:
                    stats[stat_name] = f"Error: {e}"

        return stats

    def _get_stats_mock(self) -> Dict[str, Any]:
        """Get mock mode statistics."""
        return {
            "connected": False,
            "mode": "mock",
            "total_nodes": len(self.mock_nodes),
            "concept_nodes": len(self.mock_nodes),
            "total_relationships": len(self.mock_relationships),
        }
