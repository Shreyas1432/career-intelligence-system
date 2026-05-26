"""
Operational memory persistence orchestration layer.

Provides deterministic persistence policy management, deduplication checks,
lifecycle sweeping, and orchestration services.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.modules.memory.repositories import MemoryRepository, MemorySummaryRepository
from src.modules.memory.schemas import (
    MemoryCreate,
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemoryType,
)
from src.modules.memory.significance import (
    RetentionDecision,
    SignificanceEvaluator,
    SignificanceScore,
)


class PersistenceEligibility(StrEnum):
    """Eligibility classification for persistence."""

    ELIGIBLE = "eligible"
    REVIEW_REQUIRED = "review_required"
    INELIGIBLE = "ineligible"


class ArchivalClassification(StrEnum):
    """Archival classification status based on age and significance."""

    ACTIVE = "active"
    STALE = "stale"
    ARCHIVE = "archive"


class PersistenceResult(BaseModel):
    """Result of a persistence evaluation and orchestration request."""

    entry: MemoryEntry | None = Field(
        default=None,
        description="The persisted memory entry schema, or None if not persisted.",
    )
    eligibility: PersistenceEligibility = Field(
        ...,
        description="Persistence eligibility status.",
    )
    significance: SignificanceScore = Field(
        ...,
        description="Significance score and evaluation details.",
    )
    explanation: str = Field(
        ...,
        description="Detailed explanation of the persistence decision.",
    )


class LifecycleSweepResult(BaseModel):
    """Results of a lifecycle cleanup sweep."""

    evaluated: int = Field(..., description="Number of entries evaluated.")
    retained: int = Field(..., description="Number of entries kept active.")
    deleted: int = Field(..., description="Number of entries cleaned up / deleted.")
    reviewed: int = Field(..., description="Number of entries flagged for review.")
    explanations: list[str] = Field(
        default_factory=list,
        description="Actions performed or recommendations during the sweep.",
    )


class PersistencePolicyManager:
    """
    Stateless manager responsible for evaluating persistence eligibility and archival status.
    """

    def is_content_profile_allowed(self, domain: MemoryDomain, memory_type: MemoryType) -> bool:
        """
        Check if a content profile is permitted for persistence.

        Profiles allowed:
        - Compressed operational summaries: MemoryType.SUMMARY
        - Architecture memory: MemoryDomain.ARCHITECTURE or MemoryDomain.CODEBASE
        - Relationship continuity memory: MemoryDomain.RELATIONSHIP
        - Retrieval-critical metadata: MemoryType.METADATA
        - Operational constraints: MemoryDomain.OPERATIONAL
        """
        return (
            domain == MemoryDomain.ARCHITECTURE
            or domain == MemoryDomain.CODEBASE
            or domain == MemoryDomain.RELATIONSHIP
            or domain == MemoryDomain.OPERATIONAL
            or memory_type == MemoryType.METADATA
            or memory_type == MemoryType.SUMMARY
        )

    def evaluate_eligibility(
        self,
        significance: SignificanceScore,
        domain: MemoryDomain,
        memory_type: MemoryType,
    ) -> tuple[PersistenceEligibility, str]:
        """
        Determine persistence eligibility and return a detailed reason.
        """
        # Validate allowed content profile
        if not self.is_content_profile_allowed(domain, memory_type):
            return (
                PersistenceEligibility.INELIGIBLE,
                f"Content profile ({domain.value}/{memory_type.value}) is not allowed for persistence.",
            )

        if significance.is_duplicate:
            return (
                PersistenceEligibility.INELIGIBLE,
                f"Persistence rejected: {significance.rejection_reason or 'Content is duplicate.'}",
            )

        if significance.retention_decision == RetentionDecision.REJECT:
            return (
                PersistenceEligibility.INELIGIBLE,
                f"Persistence rejected: {significance.rejection_reason or 'Failed significance threshold.'}",
            )

        if significance.retention_decision == RetentionDecision.REVIEW:
            return (
                PersistenceEligibility.REVIEW_REQUIRED,
                "Persistence requires review: content is borderline or needs verification.",
            )

        # RetentionDecision.RETAIN
        return (
            PersistenceEligibility.ELIGIBLE,
            "Persistence approved: high-confidence operational memory.",
        )

    def classify_archival(
        self, entry: MemoryEntry, reference_time: datetime | None = None
    ) -> ArchivalClassification:
        """
        Determine if an entry should be active, archived, or stale.

        Uses age-based and importance score thresholds.
        """
        created_at = entry.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        ref = reference_time or datetime.now(UTC)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=UTC)

        age_days = (ref - created_at).days

        if entry.importance_level == MemoryImportance.CRITICAL:
            return ArchivalClassification.ACTIVE

        if entry.importance_level == MemoryImportance.HIGH:
            return (
                ArchivalClassification.ACTIVE
                if age_days < 180
                else ArchivalClassification.ARCHIVE
            )

        if entry.importance_level == MemoryImportance.MEDIUM:
            if age_days < 90:
                return ArchivalClassification.ACTIVE
            elif age_days < 180:
                return ArchivalClassification.ARCHIVE
            else:
                return ArchivalClassification.STALE

        # MemoryImportance.LOW
        if age_days < 30:
            return ArchivalClassification.ACTIVE
        elif age_days < 90:
            return ArchivalClassification.ARCHIVE
        else:
            return ArchivalClassification.STALE


class MemoryDeduplicationCoordinator:
    """
    Coordinates Jaccard-based duplicate-safe persistence checks.
    """

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def get_historical_pool(self, domain: MemoryDomain, limit: int = 100) -> list[MemoryEntry]:
        """
        Retrieve existing entries for the given domain to perform duplicate verification.
        """
        return list(self.repository.list_entries(domain=domain, limit=limit))


class MemoryLifecycleManager:
    """
    Coordinates stale memory cleanup and periodic review checks.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        evaluator: SignificanceEvaluator,
        policy_manager: PersistencePolicyManager,
    ) -> None:
        self.repository = repository
        self.evaluator = evaluator
        self.policy_manager = policy_manager

    def run_lifecycle_sweep(
        self,
        *,
        low_staleness_days: int = 90,
        medium_staleness_days: int = 180,
        dry_run: bool = False,
    ) -> LifecycleSweepResult:
        """
        Sweep and cleanup stale memories below a high importance score threshold.

        Queries LOW and MEDIUM importance memories, checks if they exceed staleness days,
        re-evaluates their significance, and deletes rejected memories.
        """
        low_entries = self.repository.list_entries(
            importance_level=MemoryImportance.LOW, limit=1000
        )
        medium_entries = self.repository.list_entries(
            importance_level=MemoryImportance.MEDIUM, limit=1000
        )

        candidates = list(low_entries) + list(medium_entries)
        now = datetime.now(UTC)

        evaluated = 0
        retained = 0
        deleted = 0
        reviewed = 0
        explanations: list[str] = []

        for entry in candidates:
            created_at = entry.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)

            age_days = (now - created_at).days
            is_stale = False

            if entry.importance_level == MemoryImportance.LOW and age_days >= low_staleness_days:
                is_stale = True
            elif (
                entry.importance_level == MemoryImportance.MEDIUM
                and age_days >= medium_staleness_days
            ):
                is_stale = True

            if not is_stale:
                continue

            evaluated += 1
            sig = self.evaluator.evaluate_existing(entry)

            if sig.retention_decision == RetentionDecision.REJECT:
                reason = sig.rejection_reason or "Fails significance threshold on re-evaluation."
                explanations.append(
                    f"Deleted stale memory {entry.id} ({entry.domain.value}/{entry.importance_level.value}): {reason}"
                )
                if not dry_run:
                    self.repository.delete_entry(entry.id)
                deleted += 1
            elif sig.retention_decision == RetentionDecision.REVIEW:
                explanations.append(
                    f"Review recommended for stale memory {entry.id} ({entry.domain.value}/{entry.importance_level.value})"
                )
                reviewed += 1
            else:
                retained += 1

        return LifecycleSweepResult(
            evaluated=evaluated,
            retained=retained,
            deleted=deleted,
            reviewed=reviewed,
            explanations=explanations,
        )


class MemoryPersistenceService:
    """
    Main orchestration service coordinating the persistence of operational memory.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = MemoryRepository(session)
        self.summary_repository = MemorySummaryRepository(session)
        self.evaluator = SignificanceEvaluator()
        self.policy_manager = PersistencePolicyManager()
        self.deduplicator = MemoryDeduplicationCoordinator(self.repository)
        self.lifecycle_manager = MemoryLifecycleManager(
            repository=self.repository,
            evaluator=self.evaluator,
            policy_manager=self.policy_manager,
        )

    def persist(
        self,
        content: str,
        domain: MemoryDomain,
        memory_type: MemoryType,
        source: MemorySource,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PersistenceResult:
        """
        Orchestrates duplicate-safe, significance-aware persistence for a memory candidate.
        """
        pool = self.deduplicator.get_historical_pool(domain=domain)
        sig = self.evaluator.evaluate(
            content=content,
            domain=domain,
            memory_type=memory_type,
            source=source,
            existing_entries=pool,
        )
        eligibility, reason = self.policy_manager.evaluate_eligibility(
            significance=sig,
            domain=domain,
            memory_type=memory_type,
        )

        entry = None
        if eligibility == PersistenceEligibility.ELIGIBLE:
            create_schema = MemoryCreate(
                content=content,
                domain=domain,
                memory_type=memory_type,
                source=source,
                importance_level=sig.importance_level,
                importance_score=sig.importance_score,
                tags=tags or [],
                metadata=metadata or {},
            )
            entry = self.repository.create_entry(create_schema)

        return PersistenceResult(
            entry=entry,
            eligibility=eligibility,
            significance=sig,
            explanation=reason,
        )

    def cleanup_stale_memories(
        self,
        *,
        low_staleness_days: int = 90,
        medium_staleness_days: int = 180,
        dry_run: bool = False,
    ) -> LifecycleSweepResult:
        """
        Perform a sweep to clean up or review stale memories in the system.
        """
        return self.lifecycle_manager.run_lifecycle_sweep(
            low_staleness_days=low_staleness_days,
            medium_staleness_days=medium_staleness_days,
            dry_run=dry_run,
        )
