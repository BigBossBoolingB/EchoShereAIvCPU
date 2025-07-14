"""
EchoSphere AI-vCPU core system coordinator.

This module implements the main EchoSphere system that integrates all components
following the hybrid Actor-Blackboard model with proper startup/shutdown procedures.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from pykka import ActorRef

from .cognitive.workspace import WorkspaceActor, WorkspaceProxy
from .cognitive.actors import InputKSA, LogicKSA, OutputKSA
from .memory.memory_ksa import MemoryKSA
from .utils.config import Config
from .utils.logging import setup_logging


class EchoSphere:
    """
    Main EchoSphere AI-vCPU system coordinator.

    Manages the lifecycle of all cognitive components and provides
    a unified interface for interacting with the system.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the EchoSphere system.

        Args:
            config: System configuration object
        """
        self.config = config or Config()
        self.logger = logging.getLogger("echosphere.core")

        self.actor_system_started = False
        self.workspace_ref: Optional[ActorRef] = None
        self.workspace_proxy: Optional[WorkspaceProxy] = None

        self.input_ksa_ref: Optional[ActorRef] = None
        self.logic_ksa_ref: Optional[ActorRef] = None
        self.memory_ksa_ref: Optional[ActorRef] = None
        self.output_ksa_ref: Optional[ActorRef] = None

        self.is_running = False
        self.start_time: Optional[float] = None
        self.total_tasks_processed = 0

    def start(self) -> bool:
        """
        Start the EchoSphere system.

        Returns:
            True if startup successful, False otherwise
        """
        try:
            self.logger.info("Starting EchoSphere AI-vCPU...")

            setup_logging(self.config.log_level)

            if not self.actor_system_started:
                self.actor_system_started = True
                self.logger.info("Pykka actor system ready")

            self.workspace_ref = WorkspaceActor.start()
            self.workspace_proxy = WorkspaceProxy(self.workspace_ref)
            self.logger.info("Workspace actor started")

            self._start_knowledge_source_actors()

            time.sleep(0.5)

            if self._verify_system_health():
                self.is_running = True
                self.start_time = time.time()
                self.logger.info("EchoSphere system started successfully")
                return True
            else:
                self.logger.error("System health check failed")
                self.stop()
                return False

        except Exception as e:
            self.logger.error(f"Failed to start EchoSphere system: {e}")
            self.stop()
            return False

    def _start_knowledge_source_actors(self) -> None:
        """Start all Knowledge Source Actors."""
        self.memory_ksa_ref = MemoryKSA.start(
            self.workspace_ref,
            neo4j_uri=self.config.neo4j_uri,
            neo4j_username=self.config.neo4j_username,
            neo4j_password=self.config.neo4j_password,
        )
        self.logger.info("MemoryKSA started")

        self.logic_ksa_ref = LogicKSA.start("LogicKSA", self.workspace_ref)
        self.logger.info("LogicKSA started")

        self.input_ksa_ref = InputKSA.start("InputKSA", self.workspace_ref)
        self.logger.info("InputKSA started")

        self.output_ksa_ref = OutputKSA.start("OutputKSA", self.workspace_ref)
        self.logger.info("OutputKSA started")

    def _verify_system_health(self) -> bool:
        """Verify that all system components are healthy."""
        try:
            workspace_state = self.workspace_proxy.get_workspace_state()
            if not workspace_state.get("active"):
                self.logger.error("Workspace is not active")
                return False

            actors_to_check = [
                ("InputKSA", self.input_ksa_ref),
                ("LogicKSA", self.logic_ksa_ref),
                ("MemoryKSA", self.memory_ksa_ref),
                ("OutputKSA", self.output_ksa_ref),
            ]

            for actor_name, actor_ref in actors_to_check:
                if actor_ref is None:
                    self.logger.error(f"{actor_name} reference is None")
                    return False

                try:
                    status = actor_ref.ask({"action": "get_status"}, timeout=2)
                    if not status.get("is_active", False):
                        self.logger.error(f"{actor_name} is not active")
                        return False
                except Exception as e:
                    self.logger.error(f"{actor_name} health check failed: {e}")
                    return False

            self.logger.info("System health check passed")
            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def stop(self) -> None:
        """Stop the EchoSphere system."""
        try:
            self.logger.info("Stopping EchoSphere system...")
            self.is_running = False

            actors_to_stop = [
                ("OutputKSA", self.output_ksa_ref),
                ("InputKSA", self.input_ksa_ref),
                ("LogicKSA", self.logic_ksa_ref),
                ("MemoryKSA", self.memory_ksa_ref),
                ("Workspace", self.workspace_ref),
            ]

            for actor_name, actor_ref in actors_to_stop:
                if actor_ref:
                    try:
                        actor_ref.stop()
                        self.logger.debug(f"Stopped {actor_name}")
                    except Exception as e:
                        self.logger.warning(f"Error stopping {actor_name}: {e}")

            if self.actor_system_started:
                self.actor_system_started = False
                self.logger.info("Pykka actor system stopped")

            self.workspace_ref = None
            self.workspace_proxy = None
            self.input_ksa_ref = None
            self.logic_ksa_ref = None
            self.memory_ksa_ref = None
            self.output_ksa_ref = None

            self.logger.info("EchoSphere system stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def submit_task(self, task_type: str, content: Any) -> Dict[str, Any]:
        """
        Submit a task to the system.

        Args:
            task_type: Type of task to submit
            content: Task content

        Returns:
            Task submission result
        """
        if not self.is_running:
            return {"error": "System is not running"}

        if not self.input_ksa_ref:
            return {"error": "InputKSA not available"}

        try:
            result = self.input_ksa_ref.ask(
                {"action": "submit_task", "task_type": task_type, "content": content}
            )

            self.total_tasks_processed += 1
            return result

        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            return {"error": str(e)}

    def get_output_history(self) -> List[Dict[str, Any]]:
        """Get the output history from OutputKSA."""
        if not self.is_running or not self.output_ksa_ref:
            return []

        try:
            result = self.output_ksa_ref.ask({"action": "get_output_history"})
            return result.get("outputs", [])
        except Exception as e:
            self.logger.error(f"Error getting output history: {e}")
            return []

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get the current workspace state."""
        if not self.workspace_proxy:
            return {"error": "Workspace not available"}

        return self.workspace_proxy.get_workspace_state()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.is_running or not self.memory_ksa_ref:
            return {"error": "MemoryKSA not available"}

        try:
            return self.memory_ksa_ref.ask({"action": "get_memory_stats"})
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0

        stats = {
            "system": {
                "is_running": self.is_running,
                "uptime_seconds": uptime,
                "total_tasks_processed": self.total_tasks_processed,
                "actor_system_started": self.actor_system_started,
            },
            "workspace": self.get_workspace_state(),
            "memory": self.get_memory_stats(),
        }

        return stats

    def wait_for_completion(self, timeout: float = 10.0) -> bool:
        """
        Wait for current tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if tasks completed, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            workspace_state = self.get_workspace_state()
            active_tasks = workspace_state.get("active_tasks", 0)

            if active_tasks == 0:
                return True

            time.sleep(0.1)

        return False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
