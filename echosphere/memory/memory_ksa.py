"""
MemoryKSA: Knowledge Source Actor for memory operations.

This module implements the MemoryKSA that integrates VSA operations with
Neo4j graph storage to provide hybrid memory capabilities for the cognitive system.
"""

from typing import Any, Dict, List, Optional
import time
from pykka import ActorRef

from ..cognitive.actors import KnowledgeSourceActor
from ..cognitive.workspace import WorkspaceMessage, MessageType
from .vsa import VSAMemory
from .graph_store import GraphStore


class MemoryKSA(KnowledgeSourceActor):
    """
    Memory Knowledge Source Actor.

    Handles memory storage and retrieval operations using both VSA hypervectors
    for similarity search and Neo4j graph for symbolic relationships.
    Responds to queries from other KSAs and maintains the knowledge base.
    """

    def __init__(
        self,
        workspace_ref: ActorRef,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
    ):
        """
        Initialize the MemoryKSA.

        Args:
            workspace_ref: Reference to the workspace actor
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
        """
        super().__init__("MemoryKSA", workspace_ref)

        self.vsa_memory = VSAMemory(dimensions=10000, device="cpu")
        self.graph_store = GraphStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            vsa_memory=self.vsa_memory,
        )

        self.active_queries: Dict[str, Dict[str, Any]] = {}
        self.query_history: List[Dict[str, Any]] = []

        self.total_queries = 0
        self.successful_queries = 0
        self.average_query_time = 0.0
        self.query_type_stats = {}
        self.performance_history = []

    def initialize(self) -> None:
        """Initialize the MemoryKSA."""
        self.interested_message_types = {MessageType.QUERY}
        self.interested_tags = {"memory_query", "store_concept", "store_relation"}

        connected = self.graph_store.connect()
        if connected:
            self.logger.info("MemoryKSA connected to Neo4j")
        else:
            self.logger.warning("MemoryKSA running in mock mode (Neo4j not available)")

        self._initialize_chess_knowledge()

        self.logger.info("MemoryKSA initialized")

    def cleanup(self) -> None:
        """Clean up MemoryKSA resources."""
        self.graph_store.disconnect()
        self.active_queries.clear()
        self.logger.info("MemoryKSA cleaning up")

    def process_workspace_message(self, msg: WorkspaceMessage) -> None:
        """Process workspace messages."""
        if msg.msg_type == MessageType.QUERY:
            self._handle_query(msg)

    def handle_custom_message(self, message: Dict[str, Any]) -> Any:
        """Handle custom messages for MemoryKSA."""
        action = message.get("action")

        if action == "store_concept":
            return self._store_concept(message["concept"], message.get("properties"))
        elif action == "store_relation":
            return self._store_relation(
                message["subject"],
                message["relation"],
                message["object"],
                message.get("properties"),
            )
        elif action == "query_concept":
            return self._query_concept_direct(message["concept"])
        elif action == "get_memory_stats":
            return self._get_memory_stats()
        else:
            return super().handle_custom_message(message)

    def _initialize_chess_knowledge(self) -> None:
        """Initialize basic chess knowledge for the demo."""
        try:
            chess_pieces = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]

            for piece in chess_pieces:
                self.graph_store.create_concept_node(piece, {"category": "chess_piece"})

            self.graph_store.create_concept_node("Chess_Piece", {"category": "concept"})

            for piece in chess_pieces:
                self.graph_store.create_relationship(piece, "IS_A", "Chess_Piece")

            self.graph_store.create_relationship("Pawn", "CAN_PROMOTE_TO", "Queen")
            self.graph_store.create_relationship("Rook", "MOVES", "Straight_Lines")
            self.graph_store.create_relationship("Knight", "MOVES", "L_Shape")

            self.logger.info("Initialized chess knowledge base")

        except Exception as e:
            self.logger.error(f"Failed to initialize chess knowledge: {e}")

    def _handle_query(self, msg: WorkspaceMessage) -> None:
        """Handle a query message from the workspace."""
        query_content = msg.content
        query_id = query_content.get("query_id")
        query_type = query_content.get("query_type")

        if not query_id or not query_type:
            self.logger.warning("Received malformed query message")
            return

        start_time = time.time()
        self.active_queries[query_id] = {
            "query_type": query_type,
            "content": query_content,
            "start_time": start_time,
            "status": "processing",
        }

        self.total_queries += 1

        try:
            if query_type == "concept_analysis":
                result = self._process_concept_analysis_query(query_content)
            elif query_type == "similarity_search":
                result = self._process_similarity_search_query(query_content)
            elif query_type == "relationship_search":
                result = self._process_relationship_search_query(query_content)
            else:
                result = {"error": f"Unknown query type: {query_type}"}

            end_time = time.time()
            query_time = end_time - start_time
            self._update_performance_metrics(query_time)
            self._track_query_type_performance(query_type, query_time)

            result_content = {
                "query_id": query_id,
                "query_type": query_type,
                "result": result,
                "query_time": query_time,
                "timestamp": end_time,
            }
            result_content.update(result)  # Merge result fields into top level

            self.post_to_workspace(
                msg_type=MessageType.RESULT,
                content=result_content,
                priority=2,
                tags={"memory", "query_result"},
            )

            self.active_queries[query_id]["status"] = "completed"
            self.active_queries[query_id]["result"] = result
            self.successful_queries += 1

            self.logger.info(f"Completed query {query_id} in {query_time:.3f}s")

        except Exception as e:
            self.logger.error(f"Error processing query {query_id}: {e}")

            error_result = {
                "query_id": query_id,
                "query_type": query_type,
                "error": str(e),
                "timestamp": time.time(),
            }

            self.post_to_workspace(
                msg_type=MessageType.RESULT,
                content=error_result,
                priority=2,
                tags={"memory", "query_error"},
            )

            self.active_queries[query_id]["status"] = "error"
            self.active_queries[query_id]["error"] = str(e)

    def _process_concept_analysis_query(
        self, query_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a concept analysis query."""
        concept = query_content.get("concept")
        if not concept:
            return {"error": "No concept specified in query"}

        similar_concepts = self.graph_store.find_similar_concepts(
            concept, threshold=0.5, limit=5
        )

        relationships = self.graph_store.find_relationships(concept)

        result = {
            "concept": concept,
            "similar_concepts": [name for name, score in similar_concepts],
            "similarity_scores": {name: score for name, score in similar_concepts},
            "relations": relationships,
        }

        return result

    def _process_similarity_search_query(
        self, query_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a similarity search query."""
        concept = query_content.get("concept")
        threshold = query_content.get("threshold", 0.7)
        limit = query_content.get("limit", 5)

        if not concept:
            return {"error": "No concept specified in query"}

        similar_concepts = self.graph_store.find_similar_concepts(
            concept, threshold=threshold, limit=limit
        )

        return {
            "concept": concept,
            "similar_concepts": similar_concepts,
            "threshold": threshold,
            "limit": limit,
        }

    def _process_relationship_search_query(
        self, query_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a relationship search query."""
        concept = query_content.get("concept")
        relation_type = query_content.get("relation_type")
        direction = query_content.get("direction", "both")

        if not concept:
            return {"error": "No concept specified in query"}

        relationships = self.graph_store.find_relationships(
            concept, relation_type=relation_type, direction=direction
        )

        return {
            "concept": concept,
            "relationships": relationships,
            "relation_type": relation_type,
            "direction": direction,
        }

    def _store_concept(
        self, concept: str, properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a concept in memory."""
        try:
            node = self.graph_store.create_concept_node(concept, properties)
            return {"status": "success", "concept": concept, "node_id": node.id}
        except Exception as e:
            return {"status": "error", "concept": concept, "error": str(e)}

    def _store_relation(
        self,
        subject: str,
        relation: str,
        object_: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a relationship in memory."""
        try:
            relationship = self.graph_store.create_relationship(
                subject, relation, object_, properties
            )
            return {
                "status": "success",
                "subject": subject,
                "relation": relation,
                "object": object_,
                "relationship_id": relationship.id,
            }
        except Exception as e:
            return {
                "status": "error",
                "subject": subject,
                "relation": relation,
                "object": object_,
                "error": str(e),
            }

    def _query_concept_direct(self, concept: str) -> Dict[str, Any]:
        """Direct concept query (not through workspace)."""
        try:
            node = self.graph_store.find_concept(concept)
            if not node:
                return {"status": "not_found", "concept": concept}

            similar_concepts = self.graph_store.find_similar_concepts(concept)
            relationships = self.graph_store.find_relationships(concept)

            return {
                "status": "found",
                "concept": concept,
                "node_id": node.id,
                "properties": node.properties,
                "similar_concepts": similar_concepts,
                "relationships": relationships,
            }

        except Exception as e:
            return {"status": "error", "concept": concept, "error": str(e)}

    def _update_performance_metrics(self, query_time: float) -> None:
        """Update performance metrics with improved responsiveness."""
        alpha = min(0.3, 1.0 / max(1, self.total_queries))
        if self.average_query_time == 0.0:
            self.average_query_time = query_time
        else:
            self.average_query_time = (
                alpha * query_time + (1 - alpha) * self.average_query_time
            )

    def _track_query_type_performance(
        self, query_type: str, query_time: float
    ) -> None:
        """Track performance metrics by query type."""
        if query_type not in self.query_type_stats:
            self.query_type_stats[query_type] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }

        stats = self.query_type_stats[query_type]
        stats["count"] += 1
        stats["total_time"] += query_time
        stats["average_time"] = stats["total_time"] / stats["count"]
        stats["min_time"] = min(stats["min_time"], query_time)
        stats["max_time"] = max(stats["max_time"], query_time)

        self.performance_history.append({
            "timestamp": time.time(),
            "query_type": query_type,
            "query_time": query_time
        })

        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        vsa_stats = self.vsa_memory.get_memory_stats()
        graph_stats = self.graph_store.get_stats()

        return {
            "vsa_memory": vsa_stats,
            "graph_store": graph_stats,
            "query_stats": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "success_rate": (
                    self.successful_queries / max(1, self.total_queries)
                ) * 100,
                "average_query_time": self.average_query_time,
                "active_queries": len(self.active_queries),
                "query_type_performance": self.query_type_stats,
                "recent_performance_samples": len(self.performance_history),
            },
        }
