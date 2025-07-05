"""
Integration tests for the complete cognitive flow.
"""

import pytest
import time
from echosphere.core import EchoSphere


class TestCognitiveFlow:
    """Test cases for end-to-end cognitive processing."""

    def test_chess_analysis_flow(self):
        """Test the complete chess piece analysis flow."""
        with EchoSphere() as echo:
            time.sleep(1)

            result = echo.submit_task("analyze", "Pawn")
            assert "error" not in result

            time.sleep(2)  # Allow processing time

            outputs = echo.get_output_history()
            assert len(outputs) > 0, "No outputs generated"

            outputs = echo.get_output_history()
            assert len(outputs) > 0

            analysis = outputs[-1]["analysis"]
            assert "Pawn" in analysis
            assert any(
                keyword in analysis
                for keyword in ["Chess_Piece", "relation", "similar"]
            )

    def test_multiple_concepts(self):
        """Test processing multiple chess concepts."""
        concepts = ["Queen", "Knight", "Rook"]

        with EchoSphere() as echo:
            time.sleep(1)

            for concept in concepts:
                result = echo.submit_task("analyze", concept)
                assert "error" not in result

            time.sleep(3)  # Allow processing time for multiple concepts

            outputs = echo.get_output_history()
            assert len(outputs) >= len(concepts)

            processed_concepts = [output["concept"] for output in outputs]
            for concept in concepts:
                assert concept in processed_concepts

    def test_memory_system_integration(self):
        """Test VSA and graph store integration."""
        with EchoSphere() as echo:
            time.sleep(1)

            echo.submit_task("analyze", "Bishop")
            time.sleep(2)  # Allow processing time

            memory_stats = echo.get_memory_stats()
            assert "vsa_memory" in memory_stats
            assert "graph_store" in memory_stats

            vsa_stats = memory_stats["vsa_memory"]
            assert vsa_stats["total_concepts"] > 0
            assert vsa_stats["dimensions"] == 10000

            graph_stats = memory_stats["graph_store"]
            assert graph_stats["total_nodes"] > 0
