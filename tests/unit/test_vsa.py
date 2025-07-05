"""
Unit tests for Vector Symbolic Architecture (VSA) implementation.
"""

import pytest
import torch
from echosphere.memory.vsa import VSAMemory, HyperVector


class TestVSAMemory:
    """Test cases for VSAMemory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vsa = VSAMemory(dimensions=1000, device="cpu", vsa_type="BSC")

    def test_initialization(self):
        """Test VSA memory initialization."""
        assert self.vsa.dimensions == 1000
        assert self.vsa.device == torch.device("cpu")
        assert self.vsa.vsa_type == "BSC"
        assert len(self.vsa.codebook) == 6  # Base concepts

    def test_encode_concept(self):
        """Test concept encoding."""
        concept = "test_concept"
        hypervector = self.vsa.encode_concept(concept)

        assert isinstance(hypervector, HyperVector)
        assert hypervector.concept == concept
        assert hypervector.vector.shape == (1000,)
        assert concept in self.vsa.concept_vectors

    def test_similarity(self):
        """Test similarity computation."""
        vec1 = self.vsa.encode_concept("concept1")
        vec2 = self.vsa.encode_concept("concept2")

        self_sim = self.vsa.similarity(vec1, vec1)
        assert abs(self_sim - 1.0) < 0.01

        cross_sim = self.vsa.similarity(vec1, vec2)
        assert abs(cross_sim) < 0.5

    def test_bundle_operation(self):
        """Test bundling operation."""
        vec1 = self.vsa.encode_concept("concept1")
        vec2 = self.vsa.encode_concept("concept2")

        bundled = self.vsa.bundle([vec1, vec2])
        assert bundled.shape == (1000,)

        sim1 = self.vsa.similarity(bundled, vec1)
        sim2 = self.vsa.similarity(bundled, vec2)
        assert sim1 > 0.3
        assert sim2 > 0.3

    def test_bind_operation(self):
        """Test binding operation."""
        vec1 = self.vsa.encode_concept("concept1")
        vec2 = self.vsa.encode_concept("concept2")

        bound = self.vsa.bind(vec1, vec2)
        assert bound.shape == (1000,)

        sim1 = self.vsa.similarity(bound, vec1)
        sim2 = self.vsa.similarity(bound, vec2)
        assert abs(sim1) < 0.3
        assert abs(sim2) < 0.3

    def test_find_similar(self):
        """Test similarity search."""
        concept1 = self.vsa.encode_concept("test1")
        concept2 = self.vsa.encode_concept("test2")

        results = self.vsa.find_similar(concept1, threshold=0.9)
        assert len(results) >= 1
        assert results[0][0] == "test1"
        assert results[0][1] > 0.9

    def test_memory_stats(self):
        """Test memory statistics."""
        self.vsa.encode_concept("test1")
        self.vsa.encode_concept("test2")

        stats = self.vsa.get_memory_stats()
        assert stats["total_concepts"] >= 2
        assert stats["dimensions"] == 1000
        assert stats["vsa_type"] == "BSC"
