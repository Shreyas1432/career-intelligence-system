"""
Operational memory ranking layer.

Provides relevance scorers, freshness scorers, diversity balancers, and
ranking orchestration services.
"""

import math
from datetime import UTC, datetime

from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemoryRetrievalResult,
    MemoryType,
)


class OperationalRelevanceScorer:
    """
    Scores candidate memory entries for metadata-based operational relevance.
    """

    def score(self, entry: MemoryEntry) -> float:
        """
        Evaluate and return an operational relevance score in [0.0, 1.0].
        """
        base = entry.importance_score
        boost = 0.0

        # Bounded domain boosts
        if entry.domain in (MemoryDomain.ARCHITECTURE, MemoryDomain.RELATIONSHIP):
            boost += 0.20
        elif entry.domain == MemoryDomain.RETRIEVAL:
            boost += 0.15

        # Structural type boosts
        if entry.memory_type == MemoryType.DECISION:
            boost += 0.20
        elif entry.memory_type == MemoryType.SUMMARY:
            boost += 0.10

        # Keyword constraint boosts
        content_lower = entry.content.lower()
        if any(w in content_lower for w in ["constraint", "deadline", "blocker", "milestone"]):
            boost += 0.15

        return min(1.0, base + boost)


class FreshnessScorer:
    """
    Evaluates memory recency using exponential age decay.
    """

    def score(self, entry: MemoryEntry, reference_time: datetime | None = None) -> float:
        """
        Evaluate and return a recency score using exponential decay.
        """
        ref = reference_time or datetime.now(UTC)
        created_at = entry.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=UTC)

        age_days = (ref - created_at).days
        return math.exp(-max(0, age_days) / 180.0)


class DiversityBalancer:
    """
    Balances candidate diversity to prevent cluster domain congestion.
    """

    def balance(
        self, candidates: list[tuple[MemoryRetrievalResult, float]], limit: int
    ) -> list[MemoryRetrievalResult]:
        """
        Selects a diverse subset of candidates using domain-count penalization.
        """
        selected: list[MemoryRetrievalResult] = []
        domain_counts: dict[MemoryDomain, int] = {}

        remaining = list(candidates)
        while len(selected) < limit and remaining:
            best_idx = -1
            best_score = -float("inf")
            for idx, (res, base_score) in enumerate(remaining):
                domain = res.entry.domain
                count = domain_counts.get(domain, 0)
                # Multiplicative penalty for repetitive domains
                penalty = 0.7 ** count
                score = base_score * penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx != -1:
                res, _ = remaining.pop(best_idx)
                selected.append(res)
                domain = res.entry.domain
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            else:
                break

        return selected


class RetrievalRankingService:
    """
    Orchestrates metadata, recency, and diversity reranking for retrieved memory context.
    """

    def __init__(
        self,
        relevance_scorer: OperationalRelevanceScorer | None = None,
        freshness_scorer: FreshnessScorer | None = None,
        diversity_balancer: DiversityBalancer | None = None,
    ) -> None:
        self.relevance_scorer = relevance_scorer or OperationalRelevanceScorer()
        self.freshness_scorer = freshness_scorer or FreshnessScorer()
        self.diversity_balancer = diversity_balancer or DiversityBalancer()

    def rerank(
        self,
        results: list[MemoryRetrievalResult],
        *,
        limit: int = 5,
        similarity_weight: float = 0.4,
        relevance_weight: float = 0.4,
        freshness_weight: float = 0.2,
        reference_time: datetime | None = None,
        suppress_duplicates: bool = True,
        duplicate_threshold: float = 0.85,
    ) -> list[MemoryRetrievalResult]:
        """
        Rerank similarity matches using weighted blended scoring and duplicate suppression.
        """
        if not results:
            return []

        if suppress_duplicates:
            results = self._suppress_duplicates(results, duplicate_threshold)

        candidates = []
        for res in results:
            rel_score = self.relevance_scorer.score(res.entry)
            fresh_score = self.freshness_scorer.score(res.entry, reference_time=reference_time)

            blended = (
                (similarity_weight * res.similarity_score)
                + (relevance_weight * rel_score)
                + (freshness_weight * fresh_score)
            )
            candidates.append((res, blended))

        candidates.sort(key=lambda x: x[1], reverse=True)

        return self.diversity_balancer.balance(candidates, limit)

    def _suppress_duplicates(
        self, results: list[MemoryRetrievalResult], threshold: float
    ) -> list[MemoryRetrievalResult]:
        unique_results: list[MemoryRetrievalResult] = []
        for res in results:
            is_dup = False
            # Deduplicate same UUID
            for u_res in unique_results:
                if u_res.entry.id == res.entry.id:
                    is_dup = True
                    break
            if is_dup:
                continue

            # Deduplicate content similarity
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



__all__ = [
    "DiversityBalancer",
    "FreshnessScorer",
    "OperationalRelevanceScorer",
    "RetrievalRankingService",
]
