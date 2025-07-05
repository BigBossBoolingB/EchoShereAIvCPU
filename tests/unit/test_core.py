"""
Unit tests for EchoSphere core system.
"""

import pytest
from echosphere.core import EchoSphere
from echosphere.utils.config import Config


class TestEchoSphere:
    """Test cases for EchoSphere core system."""

    def test_initialization(self):
        """Test EchoSphere initialization."""
        config = Config()
        echo = EchoSphere(config)

        assert echo.config == config
        assert not echo.is_running
        assert echo.workspace_ref is None

    def test_system_lifecycle(self):
        """Test system start and stop."""
        echo = EchoSphere()

        success = echo.start()
        assert success
        assert echo.is_running
        assert echo.workspace_ref is not None

        echo.stop()
        assert not echo.is_running
        assert echo.workspace_ref is None

    def test_context_manager(self):
        """Test context manager interface."""
        with EchoSphere() as echo:
            assert echo.is_running

            result = echo.submit_task("analyze", "test_concept")
            assert "error" not in result

        assert not echo.is_running

    def test_system_stats(self):
        """Test system statistics."""
        with EchoSphere() as echo:
            stats = echo.get_system_stats()

            assert "system" in stats
            assert "workspace" in stats
            assert "memory" in stats
            assert stats["system"]["is_running"]
