"""
Vector Symbolic Architecture (VSA) implementation using torchhd.

This module provides the core VSA operations for hypervector manipulation:
- Bundling (⊕): Element-wise vector addition for unordered sets
- Binding (⊗): Circular convolution for ordered associations
- Similarity search and cleanup operations
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class HyperVector:
    """Represents a hypervector with metadata."""

    vector: torch.Tensor
    concept: str
    timestamp: float
    similarity_threshold: float = 0.7


class VSAMemory:
    """
    Vector Symbolic Architecture memory system using torchhd.

    Implements the core VSA operations for concept encoding, binding,
    bundling, and similarity-based retrieval.
    """

    def __init__(
        self,
        dimensions: int = 10000,
        device: str = "cpu",
        vsa_type: str = "BSC",  # Binary Spatter Code
    ):
        """
        Initialize VSA memory system.

        Args:
            dimensions: Dimensionality of hypervectors
            device: PyTorch device ('cpu' or 'cuda')
            vsa_type: Type of VSA encoding (BSC, HRR, FHRR, etc.)
        """
        self.dimensions = dimensions
        self.device = torch.device(device)
        self.vsa_type = vsa_type

        self.supported_types = ["BSC", "HRR", "FHRR"]
        if vsa_type not in self.supported_types:
            raise ValueError(
                f"Unsupported VSA type: {vsa_type}. Supported: {self.supported_types}"
            )

        self.concept_vectors: Dict[str, HyperVector] = {}
        self.codebook: Dict[str, torch.Tensor] = {}

        self._init_base_vectors()

    def _generate_random_vector(self) -> torch.Tensor:
        """Generate a random hypervector based on VSA type."""
        if self.vsa_type == "BSC":
            return torch.randint(0, 2, (self.dimensions,), device=self.device) * 2 - 1
        elif self.vsa_type in ["HRR", "FHRR"]:
            vector = torch.randn(self.dimensions, device=self.device)
            return F.normalize(vector, p=2, dim=0)
        else:
            return F.normalize(
                torch.randn(self.dimensions, device=self.device), p=2, dim=0
            )

    def _init_base_vectors(self) -> None:
        """Initialize base vectors for common VSA operations."""
        base_concepts = ["ENTITY", "RELATION", "PROPERTY", "ACTION", "LOCATION", "TIME"]

        for concept in base_concepts:
            vector = self._generate_random_vector()
            self.codebook[concept] = vector

    def encode_concept(self, concept: str, force_new: bool = False) -> HyperVector:
        """
        Encode a concept as a hypervector.

        Args:
            concept: String representation of the concept
            force_new: If True, create new vector even if concept exists

        Returns:
            HyperVector representing the concept
        """
        if concept in self.concept_vectors and not force_new:
            return self.concept_vectors[concept]

        vector = self._generate_random_vector()

        hypervector = HyperVector(
            vector=vector,
            concept=concept,
            timestamp=torch.tensor(0.0),  # Will be set by calling code
            similarity_threshold=0.7,
        )

        self.concept_vectors[concept] = hypervector
        return hypervector

    def bundle(self, vectors: List[Union[HyperVector, torch.Tensor]]) -> torch.Tensor:
        """
        Bundle (⊕) multiple vectors using element-wise addition.

        Creates an unordered set representation where the result is
        similar to all constituent vectors.

        Args:
            vectors: List of HyperVector objects or tensors to bundle

        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")

        tensor_list = []
        for v in vectors:
            if isinstance(v, HyperVector):
                tensor_list.append(v.vector)
            else:
                tensor_list.append(v)

        result = torch.stack(tensor_list).sum(dim=0)

        if self.vsa_type == "BSC":
            result = torch.sign(result)
            zero_mask = result == 0
        if zero_mask.any():
            result[zero_mask] = (
                torch.randint(0, 2, (zero_mask.sum().item(),), device=self.device) * 2
                - 1
            )
        elif self.vsa_type in ["HRR", "FHRR"]:
            result = F.normalize(result, p=2, dim=0)

        return result

    def bind(
        self,
        vector_a: Union[HyperVector, torch.Tensor],
        vector_b: Union[HyperVector, torch.Tensor],
    ) -> torch.Tensor:
        """
        Bind (⊗) two vectors using circular convolution.

        Creates ordered associations like key-value pairs where the
        result is dissimilar to inputs but preserves relational similarity.

        Args:
            vector_a: First vector to bind
            vector_b: Second vector to bind

        Returns:
            Bound hypervector
        """
        tensor_a = vector_a.vector if isinstance(vector_a, HyperVector) else vector_a
        tensor_b = vector_b.vector if isinstance(vector_b, HyperVector) else vector_b

        if self.vsa_type == "BSC":
            return tensor_a * tensor_b
        elif self.vsa_type in ["HRR", "FHRR"]:
            fft_a = torch.fft.fft(tensor_a.float())
            fft_b = torch.fft.fft(tensor_b.float())
            bound = torch.fft.ifft(fft_a * fft_b).real
            if self.vsa_type == "HRR":
                bound = F.normalize(bound, p=2, dim=0)
            return bound
        else:
            return tensor_a * tensor_b

    def unbind(
        self, bound_vector: torch.Tensor, key_vector: Union[HyperVector, torch.Tensor]
    ) -> torch.Tensor:
        """
        Unbind a vector to retrieve the associated value.

        Args:
            bound_vector: Previously bound vector
            key_vector: Key vector used in original binding

        Returns:
            Approximation of the original value vector
        """
        key_tensor = (
            key_vector.vector if isinstance(key_vector, HyperVector) else key_vector
        )
        if self.vsa_type == "BSC":
            return bound_vector * key_tensor
        elif self.vsa_type in ["HRR", "FHRR"]:
            fft_bound = torch.fft.fft(bound_vector.float())
            fft_key_conj = torch.fft.fft(key_tensor.float()).conj()
            unbound = torch.fft.ifft(fft_bound * fft_key_conj).real
            if self.vsa_type == "HRR":
                unbound = F.normalize(unbound, p=2, dim=0)
            return unbound
        else:
            return bound_vector * key_tensor

    def similarity(
        self,
        vector_a: Union[HyperVector, torch.Tensor],
        vector_b: Union[HyperVector, torch.Tensor],
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vector_a: First vector
            vector_b: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        tensor_a = vector_a.vector if isinstance(vector_a, HyperVector) else vector_a
        tensor_b = vector_b.vector if isinstance(vector_b, HyperVector) else vector_b

        # Ensure tensors are floating point for cosine similarity
        tensor_a = tensor_a.float()
        tensor_b = tensor_b.float()

        similarity_score = torch.cosine_similarity(
            tensor_a.flatten(), tensor_b.flatten(), dim=0
        )

        return float(similarity_score.item())

    def find_similar(
        self,
        query_vector: Union[HyperVector, torch.Tensor],
        threshold: float = 0.7,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find concepts similar to the query vector.

        Args:
            query_vector: Vector to search for similarities
            threshold: Minimum similarity threshold
            top_k: Maximum number of results to return

        Returns:
            List of (concept_name, similarity_score) tuples
        """
        query_tensor = (
            query_vector.vector
            if isinstance(query_vector, HyperVector)
            else query_vector
        )

        similarities = []
        for concept, hypervector in self.concept_vectors.items():
            sim_score = self.similarity(query_tensor, hypervector.vector)
            if sim_score >= threshold:
                similarities.append((concept, sim_score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cleanup(
        self, noisy_vector: torch.Tensor, candidates: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Clean up a noisy vector by finding the closest stored concept.

        Args:
            noisy_vector: Vector to clean up
            candidates: Optional list of candidate concepts to consider

        Returns:
            Tuple of (best_match_concept, similarity_score)
        """
        if candidates is None:
            candidates = list(self.concept_vectors.keys())

        best_match = None
        best_similarity = -1.0

        for concept in candidates:
            if concept in self.concept_vectors:
                sim = self.similarity(
                    noisy_vector, self.concept_vectors[concept].vector
                )
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = concept

        return best_match, best_similarity

    def create_relation(
        self, subject: str, relation: str, object_: str
    ) -> torch.Tensor:
        """
        Create a relational hypervector representing subject-relation-object.

        Args:
            subject: Subject concept
            relation: Relation type
            object_: Object concept

        Returns:
            Hypervector representing the relation
        """
        subj_vec = self.encode_concept(subject)
        rel_vec = self.encode_concept(relation)
        obj_vec = self.encode_concept(object_)

        relation_base = self.codebook["RELATION"]
        triple = self.bind(self.bind(subj_vec.vector, rel_vec.vector), obj_vec.vector)

        return self.bind(relation_base, triple)

    def query_relation(
        self, relation_vector: torch.Tensor, query_type: str, known_value: str
    ) -> List[Tuple[str, float]]:
        """
        Query a relation to find missing components.

        Args:
            relation_vector: Stored relation vector
            query_type: Type of query ('subject', 'relation', 'object')
            known_value: Known component value

        Returns:
            List of (candidate, similarity) tuples
        """
        known_vec = self.encode_concept(known_value)

        relation_base = self.codebook["RELATION"]
        triple = self.unbind(relation_vector, relation_base)

        return self.find_similar(triple, threshold=0.5)

    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the current memory state."""
        return {
            "total_concepts": len(self.concept_vectors),
            "dimensions": self.dimensions,
            "device": str(self.device),
            "vsa_type": self.vsa_type,
            "memory_usage_mb": sum(
                hv.vector.element_size() * hv.vector.nelement()
                for hv in self.concept_vectors.values()
            )
            / (1024 * 1024),
        }
