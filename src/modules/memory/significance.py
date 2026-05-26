"""
Memory significance evaluation layer.

Provides deterministic, heuristic-based scoring and filtering for operational
memory entries. All decisions are explainable and rule-driven — no LLM inference.

Components
----------
- SignificanceScore       : Output schema for a single evaluation result
- MemoryImportanceScorer  : Computes quantitative importance scores (0.0-1.0)
- NoiseFilter             : Rejects transcripts, temp-debug blobs, and low-signal content
- RetentionPolicyManager  : Classifies entries as RETAIN / REVIEW / REJECT
- SignificanceEvaluator   : Orchestrates the above into a single evaluation call
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemoryType,
)

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class RetentionDecision(StrEnum):
    """Classification outcome from the retention policy."""
    RETAIN = "retain"     # High-value; persist without review
    REVIEW = "review"     # Borderline; human verification recommended
    REJECT = "reject"     # Low-value or noisy; should not be stored


class SignificanceScore(BaseModel):
    """
    Immutable evaluation result for a single memory entry candidate.

    Fields are read-only after construction; the layer never mutates entries.
    """

    importance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Computed quantitative significance score (0.0 = noise, 1.0 = critical)",
    )
    importance_level: MemoryImportance = Field(
        ...,
        description="Qualitative band derived from importance_score",
    )
    retention_decision: RetentionDecision = Field(
        ...,
        description="RETAIN / REVIEW / REJECT classification",
    )
    is_duplicate: bool = Field(
        default=False,
        description="True when content matches an existing entry above the similarity threshold",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Human-readable reason when retention_decision is REJECT",
    )
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-factor score contributions for explainability",
    )


# ---------------------------------------------------------------------------
# Internal constants — scoring weights and keyword dictionaries
# ---------------------------------------------------------------------------

# High-signal keywords mapped to additive score boosts.
_ARCHITECTURE_SIGNALS: frozenset[str] = frozenset({
    "architecture", "refactor", "migration", "design decision", "pattern",
    "interface", "contract", "schema", "breaking change", "api", "module",
    "dependency", "coupling", "abstraction", "layering", "bounded context",
    "monorepo", "deployment", "infrastructure", "database", "index", "constraint",
})

_RELATIONSHIP_SIGNALS: frozenset[str] = frozenset({
    "recruiter", "hiring manager", "outreach", "response", "follow-up",
    "referral", "networking", "linkedin", "offer", "interview", "connection",
    "warm introduction", "contact", "preferred channel", "communication style",
    "relationship", "continuity",
})

_OPERATIONAL_SIGNALS: frozenset[str] = frozenset({
    "constraint", "deadline", "blocker", "performance", "threshold",
    "budget", "priority", "roadmap", "sprint", "milestone", "target",
    "objective", "kpi", "metric", "sla", "policy", "compliance",
})

_RETRIEVAL_SIGNALS: frozenset[str] = frozenset({
    "retrieval", "semantic", "embedding", "context window", "token",
    "similarity", "ranking", "rerank", "recall", "precision", "prompt",
    "index", "vector", "cosine", "candidate pool",
})

# Low-signal patterns that indicate noise or expendable content.
_NOISE_PATTERNS: frozenset[str] = frozenset({
    "debug", "debugging", "todo:", "fixme:", "print(", "breakpoint()",
    "temp fix", "temporary", "wip:", "placeholder", "lorem ipsum",
    "test output", "console.log", "pdb", "breakpoint",
})

_TRANSCRIPT_SIGNALS: frozenset[str] = frozenset({
    "transcript", "verbatim", "you said:", "i said:", "speaker:",
    "interviewer:", "interviewee:", "[00:", "] user:", "] assistant:",
    "session log", "raw log",
})

_FORMATTING_SIGNALS: frozenset[str] = frozenset({
    "whitespace fix", "formatting only", "renamed variable",
    "fixed typo", "reordered imports", "linting fix", "style fix",
    "trailing space", "blank line",
})

# Domain → base score contribution
_DOMAIN_BASE_SCORES: dict[MemoryDomain, float] = {
    MemoryDomain.ARCHITECTURE: 0.70,
    MemoryDomain.RELATIONSHIP: 0.60,
    MemoryDomain.RETRIEVAL:    0.55,
    MemoryDomain.CODEBASE:     0.50,
    MemoryDomain.OPERATIONAL:  0.45,
}

# MemoryType → modifier applied on top of base
_TYPE_MODIFIERS: dict[MemoryType, float] = {
    MemoryType.DECISION:  0.20,
    MemoryType.SUMMARY:   0.10,
    MemoryType.DOCUMENT:  0.05,
    MemoryType.FACT:      0.05,
    MemoryType.METADATA:  0.00,
}

# Source → trust modifier
_SOURCE_MODIFIERS: dict[MemorySource, float] = {
    MemorySource.USER:      0.10,
    MemorySource.OBSIDIAN:  0.08,
    MemorySource.SYSTEM:    0.05,
    MemorySource.INGESTION: 0.02,
    MemorySource.AI:        0.00,
}

# Score → qualitative band mapping (lower-bound inclusive)
_IMPORTANCE_BANDS: list[tuple[float, MemoryImportance]] = [
    (0.80, MemoryImportance.CRITICAL),
    (0.60, MemoryImportance.HIGH),
    (0.35, MemoryImportance.MEDIUM),
    (0.00, MemoryImportance.LOW),
]

# Retention thresholds
_RETAIN_THRESHOLD: float = 0.50
_REVIEW_THRESHOLD: float = 0.30

# Content length guard: entries longer than this are suspected blobs
_MAX_SIGNAL_LENGTH: int = 8000

# Duplicate detection: fraction of content that must be shared to flag as duplicate
_DUPLICATE_OVERLAP_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and strip a string for signal matching."""
    return text.lower().strip()


def _keyword_hit_rate(content_lower: str, keywords: frozenset[str]) -> float:
    """
    Return the fraction of ``keywords`` found in ``content_lower``.
    Capped at 1.0.
    """
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in content_lower)
    return min(1.0, hits / len(keywords))


def _band_from_score(score: float) -> MemoryImportance:
    """Map a numeric score to the appropriate qualitative importance band."""
    for threshold, band in _IMPORTANCE_BANDS:
        if score >= threshold:
            return band
    return MemoryImportance.LOW


def _word_set(text: str) -> set[str]:
    """Split content into a bag-of-words for overlap comparison."""
    return set(_normalise(text).split())


# ---------------------------------------------------------------------------
# MemoryImportanceScorer
# ---------------------------------------------------------------------------


class MemoryImportanceScorer:
    """
    Assigns a quantitative importance score (0.0-1.0) to a memory candidate.

    Scoring is additive across four independent factor groups:
    1. Domain base score         — intrinsic domain significance
    2. Memory type modifier      — decisions > summaries > facts > metadata
    3. Source trust modifier     — user > obsidian > system > ingestion > ai
    4. Content signal boost      — domain-relevant keyword density

    The final score is clamped to [0.0, 1.0].
    """

    def score(
        self,
        content: str,
        domain: MemoryDomain,
        memory_type: MemoryType,
        source: MemorySource,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute importance score and return a score_breakdown dict for explainability.

        Returns
        -------
        (score, breakdown) where score is in [0.0, 1.0] and breakdown maps
        factor names to their individual numeric contributions.
        """
        content_lower = _normalise(content)

        domain_base = _DOMAIN_BASE_SCORES.get(domain, 0.40)
        type_mod = _TYPE_MODIFIERS.get(memory_type, 0.00)
        source_mod = _SOURCE_MODIFIERS.get(source, 0.00)
        signal_boost = self._compute_signal_boost(content_lower, domain)

        raw = domain_base + type_mod + source_mod + signal_boost
        final = round(min(1.0, max(0.0, raw)), 4)

        breakdown = {
            "domain_base": domain_base,
            "type_modifier": type_mod,
            "source_modifier": source_mod,
            "signal_boost": round(signal_boost, 4),
        }
        return final, breakdown

    @staticmethod
    def _compute_signal_boost(content_lower: str, domain: MemoryDomain) -> float:
        """
        Compute the keyword-based signal boost for a specific domain.
        Max boost is 0.15; scaled by hit-rate to avoid full-score gaming.
        """
        signal_map: dict[MemoryDomain, frozenset[str]] = {
            MemoryDomain.ARCHITECTURE: _ARCHITECTURE_SIGNALS,
            MemoryDomain.RELATIONSHIP: _RELATIONSHIP_SIGNALS,
            MemoryDomain.RETRIEVAL:    _RETRIEVAL_SIGNALS,
            MemoryDomain.OPERATIONAL:  _OPERATIONAL_SIGNALS,
            MemoryDomain.CODEBASE:     _ARCHITECTURE_SIGNALS | _OPERATIONAL_SIGNALS,
        }
        keywords = signal_map.get(domain, frozenset())
        hit_rate = _keyword_hit_rate(content_lower, keywords)
        return round(hit_rate * 0.15, 4)


# ---------------------------------------------------------------------------
# NoiseFilter
# ---------------------------------------------------------------------------


class NoiseFilter:
    """
    Detects low-signal, noisy, or forbidden content patterns before scoring.

    A candidate that fails any filter is immediately flagged for rejection so
    the scorer does not waste cycles on garbage input.

    Filters (in order of cheapness):
    1. Blob guard       — content exceeding the character ceiling
    2. Transcript guard — raw conversation or session-log indicators
    3. Formatting guard — diff-only or style-only changes
    4. Noise guard      — debug artifacts, WIP markers, placeholders
    5. Repetition guard — near-identical content to an existing entry
    """

    def is_noise(self, content: str) -> tuple[bool, str | None]:
        """
        Return (is_noisy, reason).

        ``reason`` is None when the content passes all filters.
        """
        content_lower = _normalise(content)

        if self._is_blob(content):
            return True, "Content exceeds the 8,000-character signal threshold; likely a raw blob."

        if self._contains_transcript_signals(content_lower):
            return True, "Content contains transcript or raw session-log indicators."

        if self._is_formatting_only(content_lower):
            return True, "Content describes formatting-only or style-only changes; no operational signal."

        if self._contains_noise_patterns(content_lower):
            return True, "Content contains temporary debug artifacts, WIP markers, or placeholders."

        return False, None

    @staticmethod
    def _is_blob(content: str) -> bool:
        return len(content) > _MAX_SIGNAL_LENGTH

    @staticmethod
    def _contains_transcript_signals(content_lower: str) -> bool:
        return any(sig in content_lower for sig in _TRANSCRIPT_SIGNALS)

    @staticmethod
    def _is_formatting_only(content_lower: str) -> bool:
        return any(sig in content_lower for sig in _FORMATTING_SIGNALS)

    @staticmethod
    def _contains_noise_patterns(content_lower: str) -> bool:
        return any(pat in content_lower for pat in _NOISE_PATTERNS)


# ---------------------------------------------------------------------------
# RetentionPolicyManager
# ---------------------------------------------------------------------------


class RetentionPolicyManager:
    """
    Applies the platform's retention policy to produce a final classification.

    Policy thresholds
    -----------------
    - score >= 0.50  →  RETAIN  (high confidence; persist)
    - score >= 0.30  →  REVIEW  (borderline; flag for manual review)
    - score <  0.30  →  REJECT  (low value; do not persist)

    Additional overrides:
    - Any entry flagged as noise  →  REJECT regardless of score
    - Any duplicate entry         →  REJECT regardless of score
    - CRITICAL importance entries →  RETAIN regardless of threshold
    """

    def classify(
        self,
        score: float,
        importance_level: MemoryImportance,
        *,
        is_noise: bool,
        noise_reason: str | None,
        is_duplicate: bool,
    ) -> tuple[RetentionDecision, str | None]:
        """
        Return (decision, rejection_reason).

        rejection_reason is populated only when decision is REJECT.
        """
        if is_noise:
            return RetentionDecision.REJECT, noise_reason or "Noise filter triggered."

        if is_duplicate:
            return RetentionDecision.REJECT, (
                "Content is near-identical to an existing memory entry; duplicate rejected."
            )

        if importance_level == MemoryImportance.CRITICAL:
            return RetentionDecision.RETAIN, None

        if score >= _RETAIN_THRESHOLD:
            return RetentionDecision.RETAIN, None

        if score >= _REVIEW_THRESHOLD:
            return RetentionDecision.REVIEW, None

        return RetentionDecision.REJECT, (
            f"Importance score {score:.3f} is below the retention threshold "
            f"({_RETAIN_THRESHOLD:.2f}); entry does not meet quality bar."
        )


# ---------------------------------------------------------------------------
# Duplicate detection helper
# ---------------------------------------------------------------------------


def _detect_duplicate(candidate_content: str, existing_entries: list[MemoryEntry]) -> bool:
    """
    Return True if the candidate's word-set overlaps with any existing entry
    above the ``_DUPLICATE_OVERLAP_THRESHOLD``.

    Uses Jaccard similarity on bag-of-words for fast, deterministic comparison.
    Does not use embeddings — this is a pre-embedding lightweight guard.
    """
    candidate_words = _word_set(candidate_content)
    if not candidate_words:
        return False

    for entry in existing_entries:
        entry_words = _word_set(entry.content)
        if not entry_words:
            continue
        union = candidate_words | entry_words
        if not union:
            continue
        intersection = candidate_words & entry_words
        jaccard = len(intersection) / len(union)
        if jaccard >= _DUPLICATE_OVERLAP_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------------
# SignificanceEvaluator — public orchestrator
# ---------------------------------------------------------------------------


class SignificanceEvaluator:
    """
    Orchestrates noise filtering, importance scoring, duplicate detection,
    and retention classification into a single deterministic evaluation call.

    Usage::

        evaluator = SignificanceEvaluator()
        result = evaluator.evaluate(candidate_content, domain, memory_type, source)

        if result.retention_decision == RetentionDecision.RETAIN:
            repo.create_entry(MemoryCreate(..., importance_score=result.importance_score))

    The evaluator is stateless; all state is supplied as arguments.
    """

    def __init__(self) -> None:
        self._scorer = MemoryImportanceScorer()
        self._noise_filter = NoiseFilter()
        self._retention_policy = RetentionPolicyManager()

    def evaluate(
        self,
        content: str,
        domain: MemoryDomain,
        memory_type: MemoryType,
        source: MemorySource,
        *,
        existing_entries: list[MemoryEntry] | None = None,
    ) -> SignificanceScore:
        """
        Evaluate the significance of a memory candidate.

        Parameters
        ----------
        content         : Raw text of the candidate memory entry
        domain          : Bounded domain for scoring context
        memory_type     : Structural type of the entry
        source          : Origin of the memory
        existing_entries: Optional list of already-persisted entries for duplicate detection

        Returns
        -------
        SignificanceScore with importance_score, importance_level, retention_decision,
        is_duplicate, rejection_reason, and score_breakdown.
        """
        # 1. Noise filter — fast short-circuit before scoring
        is_noisy, noise_reason = self._noise_filter.is_noise(content)

        # 2. Duplicate detection — Jaccard word overlap
        entries = existing_entries or []
        is_duplicate = _detect_duplicate(content, entries)

        # 3. Importance scoring — always run for explainability even on rejects
        score, breakdown = self._scorer.score(content, domain, memory_type, source)

        # 4. Band classification
        importance_level = _band_from_score(score)

        # 5. Retention policy
        decision, rejection_reason = self._retention_policy.classify(
            score,
            importance_level,
            is_noise=is_noisy,
            noise_reason=noise_reason,
            is_duplicate=is_duplicate,
        )

        return SignificanceScore(
            importance_score=score,
            importance_level=importance_level,
            retention_decision=decision,
            is_duplicate=is_duplicate,
            rejection_reason=rejection_reason,
            score_breakdown=breakdown,
        )

    def evaluate_existing(self, entry: MemoryEntry) -> SignificanceScore:
        """
        Re-evaluate an already-persisted memory entry for retention quality.

        Useful for periodic retention sweeps. Duplicate detection is skipped
        because the entry is already in the store.
        """
        score, breakdown = self._scorer.score(
            entry.content, entry.domain, entry.memory_type, entry.source
        )
        importance_level = _band_from_score(score)
        is_noisy, noise_reason = self._noise_filter.is_noise(entry.content)

        decision, rejection_reason = self._retention_policy.classify(
            score,
            importance_level,
            is_noise=is_noisy,
            noise_reason=noise_reason,
            is_duplicate=False,
        )

        return SignificanceScore(
            importance_score=score,
            importance_level=importance_level,
            retention_decision=decision,
            is_duplicate=False,
            rejection_reason=rejection_reason,
            score_breakdown=breakdown,
        )

    def batch_evaluate(
        self,
        candidates: list[tuple[str, MemoryDomain, MemoryType, MemorySource]],
        *,
        existing_entries: list[MemoryEntry] | None = None,
    ) -> list[SignificanceScore]:
        """
        Evaluate a batch of candidates in sequence.

        Each candidate is a (content, domain, memory_type, source) tuple.
        The existing_entries pool is shared across all candidates so that
        near-duplicates within the same batch are caught when the same content
        appears more than once.
        """
        results: list[SignificanceScore] = []
        entries = list(existing_entries or [])

        for content, domain, memory_type, source in candidates:
            result = self.evaluate(
                content, domain, memory_type, source, existing_entries=entries
            )
            results.append(result)

        return results
