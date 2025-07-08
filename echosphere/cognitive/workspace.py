"""
Global Workspace Theory (GWT) implementation using Pykka actors.

This module implements the WorkspaceActor as a Facade pattern for the global workspace,
providing centralized state management and observer pattern for broadcasts.
"""

from typing import Any, Dict, List, Optional, Set
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import pykka
from pykka import ActorRef


class MessageType(Enum):
    """Types of messages that can be posted to the workspace."""

    TASK = "task"
    CONCEPT = "concept"
    RELATION = "relation"
    RESULT = "result"
    QUERY = "query"
    ATTENTION = "attention"
    STATUS = "status"


@dataclass
class WorkspaceMessage:
    """Message structure for workspace communication."""

    msg_type: MessageType
    content: Any
    source_actor: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # Higher numbers = higher priority
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkspaceActor(pykka.ThreadingActor):
    """
    Central workspace actor implementing the Global Workspace Theory.

    Acts as a Facade pattern providing a simplified interface to the complex
    internal workings of the cognitive system. Implements Observer pattern
    for broadcasting state changes to subscribed actors.
    """

    def __init__(self):
        super().__init__()
        self.messages: List[WorkspaceMessage] = []
        self.subscribers: Dict[str, ActorRef] = {}
        self.attention_focus: Optional[WorkspaceMessage] = None
        self.workspace_state: Dict[str, Any] = {}
        self.message_history: List[WorkspaceMessage] = []
        self.lock = threading.RLock()

        self.max_messages = 1000
        self.max_history = 5000
        self.attention_threshold = 2  # Minimum priority for attention

    def on_start(self) -> None:
        """Initialize the workspace when actor starts."""
        self.workspace_state = {
            "active": True,
            "total_messages": 0,
            "active_tasks": 0,
            "current_focus": None,
            "subscribers_count": 0,
        }

    def on_stop(self) -> None:
        """Clean up when actor stops."""
        with self.lock:
            self.messages.clear()
            self.subscribers.clear()
            self.workspace_state["active"] = False

    def on_receive(self, message: Dict[str, Any]) -> Any:
        """Handle incoming messages to the workspace."""
        action = message.get("action")

        if action == "post_message":
            return self._post_message(message["message"])
        elif action == "subscribe":
            return self._subscribe(message["actor_name"], message["actor_ref"])
        elif action == "unsubscribe":
            return self._unsubscribe(message["actor_name"])
        elif action == "get_messages":
            return self._get_messages(
                message.get("msg_type"), message.get("limit", 10)
            )
        elif action == "get_attention_focus":
            return self._get_attention_focus()
        elif action == "set_attention_focus":
            return self._set_attention_focus(message["message"])
        elif action == "get_workspace_state":
            return self._get_workspace_state()
        elif action == "clear_messages":
            return self._clear_messages(message.get("msg_type"))
        else:
            return {"error": f"Unknown action: {action}"}

    def _post_message(self, msg: WorkspaceMessage) -> Dict[str, Any]:
        """Post a message to the workspace and broadcast to subscribers."""
        with self.lock:
            self.messages.append(msg)
            self.message_history.append(msg)

            self.workspace_state["total_messages"] += 1
            if msg.msg_type == MessageType.TASK:
                self.workspace_state["active_tasks"] += 1

            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages :]

            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[
                    -self.max_history :
                ]

            if msg.priority >= self.attention_threshold:
                self.attention_focus = msg
                self.workspace_state["current_focus"] = msg.content

            self._broadcast_message(msg)

            return {
                "status": "success",
                "message_id": len(self.message_history) - 1,
                "timestamp": msg.timestamp,
            }

    def _subscribe(
        self, actor_name: str, actor_ref: ActorRef
    ) -> Dict[str, Any]:
        """Subscribe an actor to workspace broadcasts."""
        with self.lock:
            self.subscribers[actor_name] = actor_ref
            self.workspace_state["subscribers_count"] = len(self.subscribers)

            return {
                "status": "success",
                "subscriber_count": len(self.subscribers),
            }

    def _unsubscribe(self, actor_name: str) -> Dict[str, Any]:
        """Unsubscribe an actor from workspace broadcasts."""
        with self.lock:
            if actor_name in self.subscribers:
                del self.subscribers[actor_name]
                self.workspace_state["subscribers_count"] = len(
                    self.subscribers
                )
                return {"status": "success"}
            else:
                return {"status": "error", "message": "Actor not subscribed"}

    def _get_messages(
        self, msg_type: Optional[MessageType] = None, limit: int = 10
    ) -> List[WorkspaceMessage]:
        """Retrieve messages from the workspace."""
        with self.lock:
            messages = self.messages

            if msg_type:
                messages = [
                    msg for msg in messages if msg.msg_type == msg_type
                ]

            messages.sort(
                key=lambda x: (x.priority, x.timestamp), reverse=True
            )

            return messages[:limit]

    def _get_attention_focus(self) -> Optional[WorkspaceMessage]:
        """Get the current attention focus."""
        with self.lock:
            return self.attention_focus

    def _set_attention_focus(self, msg: WorkspaceMessage) -> Dict[str, Any]:
        """Set the attention focus to a specific message."""
        with self.lock:
            self.attention_focus = msg
            self.workspace_state["current_focus"] = msg.content

            attention_msg = WorkspaceMessage(
                msg_type=MessageType.ATTENTION,
                content={"focus_changed": msg.content},
                source_actor="WorkspaceActor",
                priority=3,
            )
            self._broadcast_message(attention_msg)

            return {"status": "success", "focus": msg.content}

    def _get_workspace_state(self) -> Dict[str, Any]:
        """Get the current workspace state."""
        with self.lock:
            return self.workspace_state.copy()

    def _clear_messages(
        self, msg_type: Optional[MessageType] = None
    ) -> Dict[str, Any]:
        """Clear messages from the workspace."""
        with self.lock:
            if msg_type:
                original_count = len(self.messages)
                self.messages = [
                    msg for msg in self.messages if msg.msg_type != msg_type
                ]
                cleared_count = original_count - len(self.messages)
            else:
                cleared_count = len(self.messages)
                self.messages.clear()
                self.attention_focus = None
                self.workspace_state["current_focus"] = None
                self.workspace_state["active_tasks"] = 0

            return {"status": "success", "cleared_count": cleared_count}

    def _broadcast_message(self, msg: WorkspaceMessage) -> None:
        """Broadcast a message to all subscribed actors."""
        broadcast = {"action": "workspace_broadcast", "message": msg}

        dead_subscribers = []
        for actor_name, actor_ref in self.subscribers.items():
            try:
                actor_ref.tell(broadcast)
            except Exception:
                dead_subscribers.append(actor_name)

        for actor_name in dead_subscribers:
            del self.subscribers[actor_name]

        if dead_subscribers:
            self.workspace_state["subscribers_count"] = len(self.subscribers)


class WorkspaceProxy:
    """
    Proxy class for easier interaction with WorkspaceActor.

    Provides a simplified interface for posting messages and querying
    the workspace without dealing with actor message passing directly.
    """

    def __init__(self, workspace_ref: ActorRef):
        self.workspace_ref = workspace_ref

    def post_message(
        self,
        msg_type: MessageType,
        content: Any,
        source_actor: str,
        priority: int = 1,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Post a message to the workspace."""
        message = WorkspaceMessage(
            msg_type=msg_type,
            content=content,
            source_actor=source_actor,
            priority=priority,
            tags=tags or set(),
            metadata=metadata or {},
        )

        return self.workspace_ref.ask(
            {"action": "post_message", "message": message}
        )

    def subscribe(
        self, actor_name: str, actor_ref: ActorRef
    ) -> Dict[str, Any]:
        """Subscribe an actor to workspace broadcasts."""
        return self.workspace_ref.ask(
            {
                "action": "subscribe",
                "actor_name": actor_name,
                "actor_ref": actor_ref,
            }
        )

    def get_messages(
        self, msg_type: Optional[MessageType] = None, limit: int = 10
    ) -> List[WorkspaceMessage]:
        """Get messages from the workspace."""
        return self.workspace_ref.ask(
            {"action": "get_messages", "msg_type": msg_type, "limit": limit}
        )

    def get_attention_focus(self) -> Optional[WorkspaceMessage]:
        """Get the current attention focus."""
        return self.workspace_ref.ask({"action": "get_attention_focus"})

    def set_attention_focus(self, msg: WorkspaceMessage) -> Dict[str, Any]:
        """Set the attention focus."""
        return self.workspace_ref.ask(
            {"action": "set_attention_focus", "message": msg}
        )

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get the current workspace state."""
        return self.workspace_ref.ask({"action": "get_workspace_state"})

    def clear_messages(
        self, msg_type: Optional[MessageType] = None
    ) -> Dict[str, Any]:
        """Clear messages from the workspace."""
        return self.workspace_ref.ask(
            {"action": "clear_messages", "msg_type": msg_type}
        )
