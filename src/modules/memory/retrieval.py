"""
Operational memory retrieval layer.

Provides metadata filters, semantic ranking coordinates, Jaccard duplicate
suppressors, context assemblers, and retrieval orchestration services.
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from sqlalchemy.orm import Session

from src.modules.memory.embeddings import EmbeddingService
from src.modules.memory.persistence import ArchivalClassification, PersistencePolicyManager
from src.modules.memory.repositories import RetrievalRepository
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEmbedding,
    MemoryEntry,
    MemoryImportance,
    MemoryRetrievalResult,
    RetrievalContext,
)


class RetrievalFilter:
    """
    Applies qualitative and metadata filtering on memory entries before similarity computations.
    """

    def __init__(self) -> None:
        self.policy_manager = PersistencePolicyManager()

    def filter_candidates(
        self,
        candidates: Sequence[tuple[MemoryEntry, MemoryEmbedding]],
        *,
        domain: MemoryDomain | None = None,
        min_importance_level: MemoryImportance | None = None,
        min_importance_score: float = 0.0,
        tag: str | None = None,
        exclude_stale: bool = True,
        reference_time: datetime | None = None,
    ) -> list[tuple[MemoryEntry, MemoryEmbedding]]:
        """
        Filters candidates on domain, minimum scores, qualitative level, stale status, and tags.
        """
        ref = reference_time or datetime.now(UTC)
        results = []

        importance_score_boundary = 0.0
        if min_importance_level:
            level_map = {
                MemoryImportance.LOW: 0.0,
                MemoryImportance.MEDIUM: 0.35,
                MemoryImportance.HIGH: 0.60,
                MemoryImportance.CRITICAL: 0.80,
            }
            importance_score_boundary = level_map.get(min_importance_level, 0.0)

        effective_min_score = max(min_importance_score, importance_score_boundary)

        for entry, emb in candidates:
            if not self._is_metadata_eligible(entry, domain, effective_min_score, tag):
                continue

            if exclude_stale:
                status = self.policy_manager.classify_archival(entry, reference_time=ref)
                if status == ArchivalClassification.STALE:
                    continue

            results.append((entry, emb))

        return results

    @staticmethod
    def _is_metadata_eligible(
        entry: MemoryEntry,
        domain: MemoryDomain | None,
        effective_min_score: float,
        tag: str | None,
    ) -> bool:
        if domain and entry.domain != domain:
            return False

        if entry.importance_score < effective_min_score:
            return False

        if tag:
            tag_lower = tag.lower()
            if not any(tag_lower in t.lower() for t in entry.tags):
                return False

        return True


class SemanticRetrievalEngine:
    """
    Ranks filtered memory candidates using vector cosine similarity.
    """

    @staticmethod
    def calculate_similarity(emb1: list[float], emb2: list[float]) -> float:
        """
        Calculate L2 normalized cosine similarity between two vectors.
        """
        if not emb1 or not emb2:
            return 0.0
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def rank_candidates(
        self,
        query_embedding: list[float],
        candidates: list[tuple[MemoryEntry, MemoryEmbedding]],
        *,
        min_similarity: float = 0.0,
        suppress_duplicates: bool = True,
        duplicate_threshold: float = 0.85,
    ) -> list[MemoryRetrievalResult]:
        """
        Rank candidates, calculate similarity scores, and suppress duplicates.
        """
        results = []
        for entry, emb in candidates:
            score = self.calculate_similarity(query_embedding, emb.embedding)
            # Clamp to valid Pydantic schema range; float32 dot-product of
            # normalised vectors can exceed ±1.0 by a tiny epsilon.
            score = max(-1.0, min(1.0, score))
            if score >= min_similarity:
                results.append(
                    MemoryRetrievalResult(
                        entry=entry,
                        similarity_score=score,
                        rerank_score=score * entry.importance_score,
                    )
                )

        results.sort(
            key=lambda x: (x.similarity_score, x.rerank_score or 0.0, x.entry.created_at),
            reverse=True,
        )

        if suppress_duplicates:
            results = self._suppress_duplicates(results, duplicate_threshold)

        return results

    @staticmethod
    def _suppress_duplicates(
        results: list[MemoryRetrievalResult], threshold: float
    ) -> list[MemoryRetrievalResult]:
        unique_results: list[MemoryRetrievalResult] = []
        for res in results:
            is_dup = False
            res_words = set(res.entry.content.lower().split())
            for u_res in unique_results:
                u_words = set(u_res.entry.content.lower().split())
                union = res_words | u_words
                if not union:
                    continue
                jaccard = len(res_words & u_words) / len(union)
                if jaccard >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique_results.append(res)
        return unique_results


class ContextAssembler:
    """
    Formats ranked matches into a single context string block.
    """

    def assemble(
        self,
        query: str,
        results: list[MemoryRetrievalResult],
        *,
        max_chars: int = 4000,
        domain_filters: list[MemoryDomain] | None = None,
    ) -> RetrievalContext:
        """
        Assemble retrieved matches into a token-efficient formatted string.
        """
        assembled_parts = []
        current_len = 0
        used_results = []

        for res in results:
            formatted = (
                f"[Domain: {res.entry.domain.value} | Type: {res.entry.memory_type.value} | "
                f"Importance: {res.entry.importance_level.value}]\n"
                f"Content: {res.entry.content}\n"
            )
            if current_len + len(formatted) + 4 > max_chars:
                break
            assembled_parts.append(formatted)
            current_len += len(formatted) + 4
            used_results.append(res)

        assembled_context = "\n---\n".join(assembled_parts).strip()
        estimated_tokens = math.ceil(len(assembled_context) / 4.0)

        return RetrievalContext(
            query=query,
            results=used_results,
            assembled_context=assembled_context,
            total_tokens=estimated_tokens,
            domain_filters=domain_filters or [],
            metadata={
                "max_chars": max_chars,
                "total_available_results": len(results),
                "total_used_results": len(used_results),
            },
        )


class MemoryRetrievalService:
    """
    Orchestration service coordinating operational memory semantic retrieval.
    """

    def __init__(self, session: Session, embedding_service: EmbeddingService) -> None:
        self.session = session
        self.embedding_service = embedding_service
        self.retrieval_repo = RetrievalRepository(session)
        self.filter = RetrievalFilter()
        self.engine = SemanticRetrievalEngine()
        self.assembler = ContextAssembler()

    def retrieve_context(
        self,
        query: str,
        *,
        domain: MemoryDomain | None = None,
        min_importance_level: MemoryImportance | None = None,
        min_importance_score: float = 0.0,
        tag: str | None = None,
        exclude_stale: bool = True,
        min_similarity: float = 0.3,
        limit: int = 5,
        max_chars: int = 4000,
        reference_time: datetime | None = None,
    ) -> RetrievalContext:
        """
        Orchestrates full semantic query retrieval and prompt context assembly.
        """
        query_embeddings = self.embedding_service.provider.generate_embeddings([query])
        if not query_embeddings:
            return RetrievalContext(
                query=query,
                results=[],
                assembled_context="",
                total_tokens=0,
                domain_filters=[domain] if domain else [],
                metadata={"error": "Query embedding generation failed."},
            )
        query_vector = query_embeddings[0]

        candidates = self.retrieval_repo.get_candidates_with_embeddings(domain=domain)

        filtered_candidates = self.filter.filter_candidates(
            candidates,
            domain=domain,
            min_importance_level=min_importance_level,
            min_importance_score=min_importance_score,
            tag=tag,
            exclude_stale=exclude_stale,
            reference_time=reference_time,
        )

        ranked_results = self.engine.rank_candidates(
            query_vector,
            filtered_candidates,
            min_similarity=min_similarity,
            suppress_duplicates=True,
        )

        limited_results = ranked_results[:limit]
        domain_filters = [domain] if domain else []

        return self.assembler.assemble(
            query,
            limited_results,
            max_chars=max_chars,
            domain_filters=domain_filters,
        )


__all__ = [
    "ContextAssembler",
    "MemoryRetrievalService",
    "RetrievalFilter",
    "SemanticRetrievalEngine",
]
