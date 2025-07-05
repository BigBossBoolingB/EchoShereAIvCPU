"""
Knowledge Source Actors (KSAs) implementing specialist cognitive modules.

This module provides the base KnowledgeSourceActor class and specific
implementations for InputKSA, LogicKSA, and OutputKSA that demonstrate
the core cognitive flow.
"""

from typing import Any, Dict, List, Optional, Set
import time
import logging
from abc import ABC, abstractmethod
import pykka
from pykka import ActorRef

from .workspace import WorkspaceMessage, MessageType, WorkspaceProxy


class KnowledgeSourceActor(pykka.ThreadingActor, ABC):
    """
    Base class for all Knowledge Source Actors (KSAs).

    Implements the common functionality for subscribing to the workspace,
    processing broadcasts, and posting messages back to the workspace.
    """

    def __init__(self, actor_name: str, workspace_ref: ActorRef):
        super().__init__()
        self.actor_name = actor_name
        self.workspace_ref = workspace_ref
        self.workspace_proxy = WorkspaceProxy(workspace_ref)
        self.logger = logging.getLogger(f"echosphere.{actor_name}")

        self.is_active = True
        self.processed_messages = 0
        self.last_activity = time.time()

        self.interested_message_types: Set[MessageType] = set()
        self.interested_tags: Set[str] = set()

    def on_start(self) -> None:
        """Initialize the actor and subscribe to workspace."""
        result = self.workspace_proxy.subscribe(self.actor_name, self.actor_ref)
        if result.get("status") == "success":
            self.logger.info(f"{self.actor_name} subscribed to workspace")
        else:
            self.logger.error(f"Failed to subscribe {self.actor_name} to workspace")

        self.initialize()

    def on_stop(self) -> None:
        """Clean up when actor stops."""
        self.is_active = False
        self.cleanup()

    def on_receive(self, message: Dict[str, Any]) -> Any:
        """Handle incoming messages."""
        action = message.get("action")

        if action == "workspace_broadcast":
            return self._handle_workspace_broadcast(message["message"])
        elif action == "get_status":
            return self._get_status()
        elif action == "set_active":
            return self._set_active(message["active"])
        else:
            return self.handle_custom_message(message)

    def _handle_workspace_broadcast(self, msg: WorkspaceMessage) -> None:
        """Handle broadcast messages from the workspace."""
        if not self._is_interested_in_message(msg):
            return

        if msg.source_actor == self.actor_name:
            return

        self.last_activity = time.time()
        self.processed_messages += 1

        try:
            self.process_workspace_message(msg)
        except Exception as e:
            self.logger.error(f"Error processing message in {self.actor_name}: {e}")

    def _is_interested_in_message(self, msg: WorkspaceMessage) -> bool:
        """Check if this actor is interested in the message."""
        if (
            self.interested_message_types
            and msg.msg_type not in self.interested_message_types
        ):
            return False

        if self.interested_tags and not (msg.tags & self.interested_tags):
            return False

        return True

    def _get_status(self) -> Dict[str, Any]:
        """Get the current status of this actor."""
        return {
            "actor_name": self.actor_name,
            "is_active": self.is_active,
            "processed_messages": self.processed_messages,
            "last_activity": self.last_activity,
            "interested_message_types": [
                mt.value for mt in self.interested_message_types
            ],
            "interested_tags": list(self.interested_tags),
        }

    def _set_active(self, active: bool) -> Dict[str, Any]:
        """Set the active state of this actor."""
        self.is_active = active
        return {"status": "success", "active": self.is_active}

    def post_to_workspace(
        self,
        msg_type: MessageType,
        content: Any,
        priority: int = 1,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post a message to the workspace."""
        return self.workspace_proxy.post_message(
            msg_type=msg_type,
            content=content,
            source_actor=self.actor_name,
            priority=priority,
            tags=tags,
            metadata=metadata,
        )

    @abstractmethod
    def initialize(self) -> None:
        """Initialize actor-specific state."""
        pass

    @abstractmethod
    def process_workspace_message(self, msg: WorkspaceMessage) -> None:
        """Process a workspace message that this actor is interested in."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up actor-specific resources."""
        pass

    def handle_custom_message(self, message: Dict[str, Any]) -> Any:
        """Handle custom messages specific to this actor type."""
        return {"error": f"Unknown message action: {message.get('action')}"}


class InputKSA(KnowledgeSourceActor):
    """
    Input Knowledge Source Actor.

    Handles external input and converts it into workspace tasks.
    In the MVD, this actor receives analysis requests and posts them as tasks.
    """

    def initialize(self) -> None:
        """Initialize the InputKSA."""
        self.interested_message_types = {MessageType.STATUS}
        self.logger.info("InputKSA initialized")

    def process_workspace_message(self, msg: WorkspaceMessage) -> None:
        """Process workspace messages."""
        if msg.msg_type == MessageType.STATUS:
            self.logger.debug(f"InputKSA received status: {msg.content}")

    def cleanup(self) -> None:
        """Clean up InputKSA resources."""
        self.logger.info("InputKSA cleaning up")

    def handle_custom_message(self, message: Dict[str, Any]) -> Any:
        """Handle custom messages for InputKSA."""
        action = message.get("action")

        if action == "submit_task":
            return self._submit_task(message["task_type"], message["content"])
        else:
            return super().handle_custom_message(message)

    def _submit_task(self, task_type: str, content: Any) -> Dict[str, Any]:
        """Submit a new task to the workspace."""
        task_content = {
            "task_type": task_type,
            "content": content,
            "submitted_by": self.actor_name,
            "timestamp": time.time(),
        }

        result = self.post_to_workspace(
            msg_type=MessageType.TASK,
            content=task_content,
            priority=2,
            tags={"input", task_type},
        )

        self.logger.info(f"Submitted task: {task_type} - {content}")
        return result


class LogicKSA(KnowledgeSourceActor):
    """
    Logic Knowledge Source Actor.

    Processes tasks and coordinates with MemoryKSA to gather information.
    Implements the reasoning logic for the cognitive system.
    """

    def initialize(self) -> None:
        """Initialize the LogicKSA."""
        self.interested_message_types = {MessageType.TASK, MessageType.RESULT}
        self.interested_tags = {"input", "memory"}
        self.pending_queries: Dict[str, Dict[str, Any]] = {}
        self.logger.info("LogicKSA initialized")

    def process_workspace_message(self, msg: WorkspaceMessage) -> None:
        """Process workspace messages."""
        if msg.msg_type == MessageType.TASK:
            self._process_task(msg)
        elif msg.msg_type == MessageType.RESULT:
            self._process_result(msg)

    def cleanup(self) -> None:
        """Clean up LogicKSA resources."""
        self.pending_queries.clear()
        self.logger.info("LogicKSA cleaning up")

    def _process_task(self, msg: WorkspaceMessage) -> None:
        """Process a task message."""
        task_content = msg.content
        task_type = task_content.get("task_type")

        if task_type == "analyze":
            self._handle_analyze_task(task_content)
        else:
            self.logger.warning(f"Unknown task type: {task_type}")

    def _handle_analyze_task(self, task_content: Dict[str, Any]) -> None:
        """Handle an analysis task."""
        concept = task_content.get("content")
        query_id = f"analyze_{concept}_{time.time()}"

        self.pending_queries[query_id] = {
            "concept": concept,
            "task_content": task_content,
            "timestamp": time.time(),
            "status": "pending",
        }

        query_content = {
            "query_id": query_id,
            "query_type": "concept_analysis",
            "concept": concept,
            "requested_by": self.actor_name,
        }

        self.post_to_workspace(
            msg_type=MessageType.QUERY,
            content=query_content,
            priority=2,
            tags={"logic", "memory_query"},
        )

        self.logger.info(f"Queried memory for concept analysis: {concept}")

    def _process_result(self, msg: WorkspaceMessage) -> None:
        """Process a result message from MemoryKSA."""
        result_content = msg.content
        query_id = result_content.get("query_id")

        if query_id in self.pending_queries:
            query_info = self.pending_queries[query_id]
            concept = query_info["concept"]

            similar_concepts = result_content.get("similar_concepts", [])
            relations = result_content.get("relations", [])

            analysis = self._generate_analysis(concept, similar_concepts, relations)

            analysis_result = {
                "original_concept": concept,
                "analysis": analysis,
                "query_id": query_id,
                "generated_by": self.actor_name,
            }

            self.post_to_workspace(
                msg_type=MessageType.RESULT,
                content=analysis_result,
                priority=2,
                tags={"logic", "analysis_complete"},
            )

            del self.pending_queries[query_id]

            self.logger.info(f"Completed analysis for: {concept}")

    def _generate_analysis(
        self, concept: str, similar_concepts: List[str], relations: List[Dict[str, str]]
    ) -> str:
        """Generate a coherent analysis from memory results."""
        analysis_parts = [f"Analysis of '{concept}':"]

        if relations:
            for relation in relations:
                rel_type = relation.get("relation", "related_to")
                target = relation.get("object", "unknown")
                analysis_parts.append(f"Found relation '{rel_type}: {target}'.")

        if similar_concepts:
            similar_list = ", ".join(f"'{sc}'" for sc in similar_concepts[:3])
            analysis_parts.append(f"Found similar concept(s): {similar_list}.")

        if not relations and not similar_concepts:
            analysis_parts.append("No significant relations or similarities found.")

        return " ".join(analysis_parts)


class OutputKSA(KnowledgeSourceActor):
    """
    Output Knowledge Source Actor.

    Handles final output generation and presentation of results.
    Monitors for completed analyses and formats them for output.
    """

    def initialize(self) -> None:
        """Initialize the OutputKSA."""
        self.interested_message_types = {MessageType.RESULT}
        self.interested_tags = {"analysis_complete"}
        self.output_history: List[Dict[str, Any]] = []
        self.logger.info("OutputKSA initialized")

    def process_workspace_message(self, msg: WorkspaceMessage) -> None:
        """Process workspace messages."""
        if msg.msg_type == MessageType.RESULT and "analysis_complete" in msg.tags:
            self._process_analysis_result(msg)

    def cleanup(self) -> None:
        """Clean up OutputKSA resources."""
        self.output_history.clear()
        self.logger.info("OutputKSA cleaning up")

    def handle_custom_message(self, message: Dict[str, Any]) -> Any:
        """Handle custom messages for OutputKSA."""
        action = message.get("action")

        if action == "get_output_history":
            return {"outputs": self.output_history}
        else:
            return super().handle_custom_message(message)

    def _process_analysis_result(self, msg: WorkspaceMessage) -> None:
        """Process an analysis result and generate output."""
        result_content = msg.content
        analysis = result_content.get("analysis", "No analysis available")
        concept = result_content.get("original_concept", "Unknown")

        output = {
            "timestamp": time.time(),
            "concept": concept,
            "analysis": analysis,
            "source": msg.source_actor,
        }

        self.output_history.append(output)

        if len(self.output_history) > 100:
            self.output_history = self.output_history[-100:]

        self.logger.info(f"OUTPUT: {analysis}")

        status_content = {
            "status": "output_generated",
            "concept": concept,
            "output": analysis,
        }

        self.post_to_workspace(
            msg_type=MessageType.STATUS,
            content=status_content,
            priority=1,
            tags={"output", "complete"},
        )
