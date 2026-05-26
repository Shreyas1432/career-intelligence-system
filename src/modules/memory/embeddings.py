"""
Operational memory embeddings layer.

Provides provider abstractions, concrete local SentenceTransformer providers,
vector normalizers, eligibility evaluators, and orchestration services.
"""

import math
from abc import ABC, abstractmethod
from typing import Any

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from src.modules.memory.repositories import EmbeddingRepository
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEmbedding,
    MemoryEntry,
    MemoryImportance,
    MemoryType,
)


class EmbeddingProvider(ABC):
    """
    Abstract base class defining the interface for embedding generation providers.
    """

    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense vector embeddings for a list of input texts.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the identifier string of the underlying embedding model.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension size of the generated vectors.
        """
        pass


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """
    Concrete embedding provider executing SentenceTransformer models locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in vector] for vector in vectors.tolist()]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingNormalizer:
    """
    Utilities for L2 normalization and dimensional validation of embedding vectors.
    """

    @staticmethod
    def normalize(vector: list[float]) -> list[float]:
        """
        Perform L2 normalization on a dense float vector.
        """
        if not vector:
            return []
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    @staticmethod
    def validate_dimension(vector: list[float], expected_dimension: int) -> bool:
        """
        Validate that the length of the vector matches the expected dimension.
        """
        return len(vector) == expected_dimension


class EmbeddingEligibilityEvaluator:
    """
    Evaluates whether a memory entry is eligible for semantic embedding generation.
    """

    def is_eligible(self, entry: MemoryEntry) -> bool:
        """
        Evaluates eligibility based on content length, importance, and profile type.
        """
        if entry.importance_level == MemoryImportance.LOW:
            return False

        if len(entry.content) > 8000:
            return False

        content_lower = entry.content.lower()
        if any(
            sig in content_lower
            for sig in ["transcript:", "speaker:", "interviewer:", "session log"]
        ):
            return False

        # Verify allowed categories
        return (
            entry.importance_level in (MemoryImportance.HIGH, MemoryImportance.CRITICAL)
            or entry.domain == MemoryDomain.ARCHITECTURE
            or entry.domain == MemoryDomain.RELATIONSHIP
            or entry.memory_type == MemoryType.DECISION
            or entry.domain == MemoryDomain.RETRIEVAL
            or entry.memory_type == MemoryType.METADATA
            or entry.memory_type == MemoryType.SUMMARY
        )


class EmbeddingService:
    """
    Service coordinating embedding generation, normalization, and database storage.
    """

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self.provider = provider or LocalSentenceTransformerProvider()
        self.normalizer = EmbeddingNormalizer()
        self.evaluator = EmbeddingEligibilityEvaluator()

    def generate_embedding(self, entry: MemoryEntry) -> MemoryEmbedding | None:
        """
        Generate, normalize, and validate the embedding vector for a single eligible entry.
        """
        if not self.evaluator.is_eligible(entry):
            return None

        embeddings = self.provider.generate_embeddings([entry.content])
        if not embeddings:
            return None

        vector = embeddings[0]
        if not self.normalizer.validate_dimension(vector, self.provider.dimension):
            return None

        normalized = self.normalizer.normalize(vector)

        return MemoryEmbedding(
            memory_id=entry.id,
            embedding=normalized,
            model_name=self.provider.model_name,
            dimension=self.provider.dimension,
        )

    def generate_embeddings_batch(self, entries: list[MemoryEntry]) -> list[MemoryEmbedding]:
        """
        Generate, normalize, and validate embeddings for a batch of eligible entries.
        """
        eligible_entries = [e for e in entries if self.evaluator.is_eligible(e)]
        if not eligible_entries:
            return []

        texts = [e.content for e in eligible_entries]
        vectors = self.provider.generate_embeddings(texts)

        results = []
        for entry, vector in zip(eligible_entries, vectors, strict=True):
            if not self.normalizer.validate_dimension(vector, self.provider.dimension):
                continue
            normalized = self.normalizer.normalize(vector)
            results.append(
                MemoryEmbedding(
                    memory_id=entry.id,
                    embedding=normalized,
                    model_name=self.provider.model_name,
                    dimension=self.provider.dimension,
                )
            )

        return results

    def orchestrate_embeddings(
        self, session: Session, entries: list[MemoryEntry]
    ) -> list[MemoryEmbedding]:
        """
        Orchestrate embedding generation and save the resulting vectors in the database.
        """
        embeddings = self.generate_embeddings_batch(entries)
        if not embeddings:
            return []

        repo = EmbeddingRepository(session)
        for emb in embeddings:
            repo.save_embedding(emb)

        return embeddings


__all__ = [
    "EmbeddingEligibilityEvaluator",
    "EmbeddingNormalizer",
    "EmbeddingProvider",
    "EmbeddingService",
    "LocalSentenceTransformerProvider",
]
