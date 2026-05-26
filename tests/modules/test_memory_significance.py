"""
Tests for src/modules/memory/significance.py

Coverage
--------
- MemoryImportanceScorer  : score ranges, factor contributions, domain/type/source sensitivity
- NoiseFilter             : each of the five guards independently
- RetentionPolicyManager  : all classification branches and overrides
- SignificanceEvaluator   : orchestrated evaluate(), evaluate_existing(), batch_evaluate()
- Duplicate detection     : Jaccard similarity threshold behaviour
- Determinism             : same input always yields same output
"""

from uuid import uuid4

import pytest

from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySource,
    MemoryType,
)
from src.modules.memory.significance import (
    MemoryImportanceScorer,
    NoiseFilter,
    RetentionDecision,
    RetentionPolicyManager,
    SignificanceEvaluator,
    SignificanceScore,
)

# ---------------------------------------------------------------------------
# Shared fixtures and factory helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> MemoryImportanceScorer:
    return MemoryImportanceScorer()


@pytest.fixture
def noise_filter() -> NoiseFilter:
    return NoiseFilter()


@pytest.fixture
def policy() -> RetentionPolicyManager:
    return RetentionPolicyManager()


@pytest.fixture
def evaluator() -> SignificanceEvaluator:
    return SignificanceEvaluator()


def _make_entry(
    content: str = "Generic operational note.",
    domain: MemoryDomain = MemoryDomain.OPERATIONAL,
    memory_type: MemoryType = MemoryType.FACT,
    source: MemorySource = MemorySource.SYSTEM,
    importance_level: MemoryImportance = MemoryImportance.MEDIUM,
    importance_score: float = 0.5,
) -> MemoryEntry:
    return MemoryEntry(
        id=uuid4(),
        content=content,
        domain=domain,
        memory_type=memory_type,
        source=source,
        importance_level=importance_level,
        importance_score=importance_score,
    )


# ---------------------------------------------------------------------------
# MemoryImportanceScorer — score range integrity
# ---------------------------------------------------------------------------


class TestMemoryImportanceScorer:
    def test_score_is_within_unit_interval(self, scorer: MemoryImportanceScorer) -> None:
        score, _ = scorer.score(
            "Architecture decision: use WAL mode for SQLite.",
            MemoryDomain.ARCHITECTURE,
            MemoryType.DECISION,
            MemorySource.USER,
        )
        assert 0.0 <= score <= 1.0

    def test_breakdown_contains_expected_keys(self, scorer: MemoryImportanceScorer) -> None:
        _, breakdown = scorer.score(
            "Some note.",
            MemoryDomain.OPERATIONAL,
            MemoryType.FACT,
            MemorySource.SYSTEM,
        )
        assert set(breakdown.keys()) == {
            "domain_base",
            "type_modifier",
            "source_modifier",
            "signal_boost",
        }

    def test_domain_base_differentiates_architecture_vs_operational(
        self, scorer: MemoryImportanceScorer
    ) -> None:
        """Architecture domain should produce a higher base score than operational."""
        arch_score, _ = scorer.score(
            "A plain note.",
            MemoryDomain.ARCHITECTURE,
            MemoryType.FACT,
            MemorySource.SYSTEM,
        )
        ops_score, _ = scorer.score(
            "A plain note.",
            MemoryDomain.OPERATIONAL,
            MemoryType.FACT,
            MemorySource.SYSTEM,
        )
        assert arch_score > ops_score

    def test_decision_type_scores_higher_than_metadata(
        self, scorer: MemoryImportanceScorer
    ) -> None:
        decision_score, _ = scorer.score(
            "A note.", MemoryDomain.OPERATIONAL, MemoryType.DECISION, MemorySource.SYSTEM
        )
        metadata_score, _ = scorer.score(
            "A note.", MemoryDomain.OPERATIONAL, MemoryType.METADATA, MemorySource.SYSTEM
        )
        assert decision_score > metadata_score

    def test_user_source_scores_higher_than_ai(self, scorer: MemoryImportanceScorer) -> None:
        user_score, _ = scorer.score(
            "A note.", MemoryDomain.OPERATIONAL, MemoryType.FACT, MemorySource.USER
        )
        ai_score, _ = scorer.score(
            "A note.", MemoryDomain.OPERATIONAL, MemoryType.FACT, MemorySource.AI
        )
        assert user_score > ai_score

    def test_architecture_keywords_boost_score(self, scorer: MemoryImportanceScorer) -> None:
        """Content with domain-relevant keywords should score higher than plain content."""
        signal_rich = (
            "Architecture decision: migrate the module to bounded context. "
            "Breaking change in the interface contract and API schema."
        )
        plain = "Some generic text without any relevant terms."

        rich_score, _ = scorer.score(
            signal_rich, MemoryDomain.ARCHITECTURE, MemoryType.FACT, MemorySource.SYSTEM
        )
        plain_score, _ = scorer.score(
            plain, MemoryDomain.ARCHITECTURE, MemoryType.FACT, MemorySource.SYSTEM
        )
        assert rich_score > plain_score

    def test_relationship_keywords_boost_score(self, scorer: MemoryImportanceScorer) -> None:
        signal_rich = (
            "Recruiter Jane at Acme prefers LinkedIn outreach. "
            "Follow-up after a positive response within 2 days."
        )
        plain = "Something happened today."

        rich_score, _ = scorer.score(
            signal_rich, MemoryDomain.RELATIONSHIP, MemoryType.FACT, MemorySource.SYSTEM
        )
        plain_score, _ = scorer.score(
            plain, MemoryDomain.RELATIONSHIP, MemoryType.FACT, MemorySource.SYSTEM
        )
        assert rich_score > plain_score

    def test_score_is_deterministic(self, scorer: MemoryImportanceScorer) -> None:
        """Same inputs must always yield the same score."""
        args = (
            "Architecture decision on module coupling.",
            MemoryDomain.ARCHITECTURE,
            MemoryType.DECISION,
            MemorySource.USER,
        )
        score_a, breakdown_a = scorer.score(*args)
        score_b, breakdown_b = scorer.score(*args)
        assert score_a == score_b
        assert breakdown_a == breakdown_b

    def test_breakdown_values_are_non_negative(self, scorer: MemoryImportanceScorer) -> None:
        _, breakdown = scorer.score(
            "Refactoring the schema abstraction layer.",
            MemoryDomain.CODEBASE,
            MemoryType.DECISION,
            MemorySource.OBSIDIAN,
        )
        assert all(v >= 0.0 for v in breakdown.values())


# ---------------------------------------------------------------------------
# NoiseFilter — individual guard coverage
# ---------------------------------------------------------------------------


class TestNoiseFilter:
    def test_clean_content_passes_all_filters(self, noise_filter: NoiseFilter) -> None:
        clean = (
            "Architecture decision: adopt WAL journal mode for SQLite to improve "
            "write concurrency under local-first conditions."
        )
        is_noisy, reason = noise_filter.is_noise(clean)
        assert is_noisy is False
        assert reason is None

    def test_blob_guard_rejects_long_content(self, noise_filter: NoiseFilter) -> None:
        blob = "x" * 8001
        is_noisy, reason = noise_filter.is_noise(blob)
        assert is_noisy is True
        assert reason is not None
        assert "blob" in reason.lower() or "threshold" in reason.lower()

    def test_transcript_guard_rejects_session_log_indicators(
        self, noise_filter: NoiseFilter
    ) -> None:
        transcript = "transcript: User asked about the refactor and I said: yes, we should do it."
        is_noisy, reason = noise_filter.is_noise(transcript)
        assert is_noisy is True
        assert reason is not None

    def test_transcript_guard_rejects_speaker_labels(self, noise_filter: NoiseFilter) -> None:
        log = "Interviewer: Tell me about your experience. Interviewee: I worked on..."
        is_noisy, _reason = noise_filter.is_noise(log)
        assert is_noisy is True

    def test_formatting_guard_rejects_style_only_changes(self, noise_filter: NoiseFilter) -> None:
        style_note = "Whitespace fix: removed trailing space from config file."
        is_noisy, reason = noise_filter.is_noise(style_note)
        assert is_noisy is True
        assert reason is not None

    def test_formatting_guard_rejects_linting_fix(self, noise_filter: NoiseFilter) -> None:
        lint_note = "Linting fix applied to resolve ruff warnings across three files."
        is_noisy, _reason = noise_filter.is_noise(lint_note)
        assert is_noisy is True

    def test_noise_guard_rejects_debug_artifacts(self, noise_filter: NoiseFilter) -> None:
        debug_note = "Added debug breakpoint() to trace the failing query path."
        is_noisy, _reason = noise_filter.is_noise(debug_note)
        assert is_noisy is True

    def test_noise_guard_rejects_wip_markers(self, noise_filter: NoiseFilter) -> None:
        wip = "WIP: not finished yet — placeholder logic for the retry handler."
        is_noisy, _reason = noise_filter.is_noise(wip)
        assert is_noisy is True

    def test_noise_guard_rejects_todo_markers(self, noise_filter: NoiseFilter) -> None:
        todo = "TODO: refactor this entire module before the release."
        is_noisy, _reason = noise_filter.is_noise(todo)
        assert is_noisy is True

    def test_content_exactly_at_length_limit_passes(self, noise_filter: NoiseFilter) -> None:
        """Content at exactly 8,000 chars should pass the blob guard."""
        edge = "a" * 8000
        is_noisy, _ = noise_filter.is_noise(edge)
        assert is_noisy is False

    def test_noise_detection_is_case_insensitive(self, noise_filter: NoiseFilter) -> None:
        upper_transcript = "TRANSCRIPT: everything said in the meeting."
        is_noisy, _ = noise_filter.is_noise(upper_transcript)
        assert is_noisy is True


# ---------------------------------------------------------------------------
# RetentionPolicyManager — classification branches
# ---------------------------------------------------------------------------


class TestRetentionPolicyManager:
    def test_noise_always_rejected_regardless_of_score(
        self, policy: RetentionPolicyManager
    ) -> None:
        decision, reason = policy.classify(
            0.95,
            MemoryImportance.CRITICAL,
            is_noise=True,
            noise_reason="Content is a raw blob.",
            is_duplicate=False,
        )
        assert decision == RetentionDecision.REJECT
        assert reason is not None

    def test_duplicate_always_rejected_regardless_of_score(
        self, policy: RetentionPolicyManager
    ) -> None:
        decision, reason = policy.classify(
            0.90,
            MemoryImportance.HIGH,
            is_noise=False,
            noise_reason=None,
            is_duplicate=True,
        )
        assert decision == RetentionDecision.REJECT
        assert reason is not None
        assert "duplicate" in reason.lower()

    def test_critical_importance_retained_regardless_of_score(
        self, policy: RetentionPolicyManager
    ) -> None:
        """CRITICAL entries bypass the score threshold and are always retained."""
        decision, reason = policy.classify(
            0.20,  # below normal retain threshold
            MemoryImportance.CRITICAL,
            is_noise=False,
            noise_reason=None,
            is_duplicate=False,
        )
        assert decision == RetentionDecision.RETAIN
        assert reason is None

    def test_high_score_retained(self, policy: RetentionPolicyManager) -> None:
        decision, reason = policy.classify(
            0.75,
            MemoryImportance.HIGH,
            is_noise=False,
            noise_reason=None,
            is_duplicate=False,
        )
        assert decision == RetentionDecision.RETAIN
        assert reason is None

    def test_borderline_score_routed_to_review(self, policy: RetentionPolicyManager) -> None:
        decision, _ = policy.classify(
            0.40,
            MemoryImportance.MEDIUM,
            is_noise=False,
            noise_reason=None,
            is_duplicate=False,
        )
        assert decision == RetentionDecision.REVIEW

    def test_low_score_rejected_with_reason(self, policy: RetentionPolicyManager) -> None:
        decision, reason = policy.classify(
            0.15,
            MemoryImportance.LOW,
            is_noise=False,
            noise_reason=None,
            is_duplicate=False,
        )
        assert decision == RetentionDecision.REJECT
        assert reason is not None
        assert "0.15" in reason or "threshold" in reason.lower()

    def test_exact_retain_boundary_is_retained(self, policy: RetentionPolicyManager) -> None:
        decision, _ = policy.classify(
            0.50,
            MemoryImportance.MEDIUM,
            is_noise=False,
            noise_reason=None,
            is_duplicate=False,
        )
        assert decision == RetentionDecision.RETAIN

    def test_exact_review_boundary_is_reviewed(self, policy: RetentionPolicyManager) -> None:
        decision, _ = policy.classify(
            0.30,
            MemoryImportance.MEDIUM,
            is_noise=False,
            noise_reason=None,
            is_duplicate=False,
        )
        assert decision == RetentionDecision.REVIEW


# ---------------------------------------------------------------------------
# SignificanceEvaluator — evaluate()
# ---------------------------------------------------------------------------


class TestSignificanceEvaluatorEvaluate:
    def test_returns_significance_score_instance(self, evaluator: SignificanceEvaluator) -> None:
        result = evaluator.evaluate(
            "Recruiter outreach follow-up strategy for LinkedIn contacts.",
            MemoryDomain.RELATIONSHIP,
            MemoryType.FACT,
            MemorySource.USER,
        )
        assert isinstance(result, SignificanceScore)

    def test_architecture_decision_is_retained(self, evaluator: SignificanceEvaluator) -> None:
        result = evaluator.evaluate(
            "Architecture decision: adopt bounded context separation for the memory module. "
            "This is a breaking change to the existing interface contract and API schema.",
            MemoryDomain.ARCHITECTURE,
            MemoryType.DECISION,
            MemorySource.USER,
        )
        assert result.retention_decision == RetentionDecision.RETAIN
        assert result.importance_score >= 0.50

    def test_recruiter_relationship_note_is_retained(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        result = evaluator.evaluate(
            "Recruiter Jane at Acme Corp prefers LinkedIn outreach. "
            "Response time is fast; follow-up after positive response within 2 days.",
            MemoryDomain.RELATIONSHIP,
            MemoryType.FACT,
            MemorySource.USER,
        )
        assert result.retention_decision == RetentionDecision.RETAIN

    def test_noise_content_is_rejected(self, evaluator: SignificanceEvaluator) -> None:
        result = evaluator.evaluate(
            "debug: added breakpoint() to trace the ORM flush.",
            MemoryDomain.OPERATIONAL,
            MemoryType.FACT,
            MemorySource.SYSTEM,
        )
        assert result.retention_decision == RetentionDecision.REJECT
        assert result.rejection_reason is not None

    def test_transcript_is_rejected(self, evaluator: SignificanceEvaluator) -> None:
        result = evaluator.evaluate(
            "Transcript: interviewer said tell me about yourself and I said I have 5 years...",
            MemoryDomain.OPERATIONAL,
            MemoryType.DOCUMENT,
            MemorySource.INGESTION,
        )
        assert result.retention_decision == RetentionDecision.REJECT

    def test_blob_is_rejected(self, evaluator: SignificanceEvaluator) -> None:
        result = evaluator.evaluate(
            "x" * 8001,
            MemoryDomain.ARCHITECTURE,
            MemoryType.DOCUMENT,
            MemorySource.INGESTION,
        )
        assert result.retention_decision == RetentionDecision.REJECT
        assert result.rejection_reason is not None

    def test_score_breakdown_is_populated(self, evaluator: SignificanceEvaluator) -> None:
        result = evaluator.evaluate(
            "Operational constraint: deployment must complete before Monday.",
            MemoryDomain.OPERATIONAL,
            MemoryType.FACT,
            MemorySource.SYSTEM,
        )
        assert "domain_base" in result.score_breakdown
        assert "type_modifier" in result.score_breakdown
        assert "source_modifier" in result.score_breakdown
        assert "signal_boost" in result.score_breakdown

    def test_importance_level_matches_score(self, evaluator: SignificanceEvaluator) -> None:
        """The qualitative band must be consistent with the numeric score."""
        result = evaluator.evaluate(
            "Codebase refactor decision for the repository abstraction layer.",
            MemoryDomain.CODEBASE,
            MemoryType.DECISION,
            MemorySource.USER,
        )
        if result.importance_score >= 0.80:
            assert result.importance_level == MemoryImportance.CRITICAL
        elif result.importance_score >= 0.60:
            assert result.importance_level == MemoryImportance.HIGH
        elif result.importance_score >= 0.35:
            assert result.importance_level == MemoryImportance.MEDIUM
        else:
            assert result.importance_level == MemoryImportance.LOW


# ---------------------------------------------------------------------------
# SignificanceEvaluator — duplicate detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    def test_near_identical_content_flagged_as_duplicate(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        content = (
            "Use WAL journal mode for SQLite to improve write concurrency "
            "under local-first conditions without blocking reads."
        )
        existing = [_make_entry(content=content)]

        result = evaluator.evaluate(
            content,  # exact same content
            MemoryDomain.ARCHITECTURE,
            MemoryType.DECISION,
            MemorySource.USER,
            existing_entries=existing,
        )
        assert result.is_duplicate is True
        assert result.retention_decision == RetentionDecision.REJECT

    def test_distinct_content_not_flagged_as_duplicate(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        existing = [
            _make_entry(content="Use WAL journal mode for SQLite write performance.")
        ]
        result = evaluator.evaluate(
            "Recruiter Jane prefers LinkedIn DMs over cold email outreach.",
            MemoryDomain.RELATIONSHIP,
            MemoryType.FACT,
            MemorySource.USER,
            existing_entries=existing,
        )
        assert result.is_duplicate is False

    def test_no_existing_entries_never_flags_duplicate(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        result = evaluator.evaluate(
            "Architecture decision: use bounded context modules.",
            MemoryDomain.ARCHITECTURE,
            MemoryType.DECISION,
            MemorySource.USER,
            existing_entries=[],
        )
        assert result.is_duplicate is False

    def test_partially_overlapping_content_not_flagged(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        """Content sharing a few common words should not cross the Jaccard threshold."""
        existing = [_make_entry(content="The database uses SQLite for persistence.")]
        result = evaluator.evaluate(
            "Recruiter at Acme Corp has accepted my LinkedIn connection request.",
            MemoryDomain.RELATIONSHIP,
            MemoryType.FACT,
            MemorySource.USER,
            existing_entries=existing,
        )
        assert result.is_duplicate is False


# ---------------------------------------------------------------------------
# SignificanceEvaluator — evaluate_existing()
# ---------------------------------------------------------------------------


class TestEvaluateExisting:
    def test_evaluate_existing_returns_significance_score(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        entry = _make_entry(
            content="Architecture decision: adopt repository pattern for data access.",
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )
        result = evaluator.evaluate_existing(entry)
        assert isinstance(result, SignificanceScore)

    def test_evaluate_existing_is_never_a_duplicate(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        entry = _make_entry(content="Some stored fact about the codebase.")
        result = evaluator.evaluate_existing(entry)
        assert result.is_duplicate is False

    def test_evaluate_existing_noisy_entry_is_rejected(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        noisy_entry = _make_entry(
            content="debug: temporary breakpoint() added for query tracing.",
            domain=MemoryDomain.OPERATIONAL,
        )
        result = evaluator.evaluate_existing(noisy_entry)
        assert result.retention_decision == RetentionDecision.REJECT


# ---------------------------------------------------------------------------
# SignificanceEvaluator — batch_evaluate()
# ---------------------------------------------------------------------------


class TestBatchEvaluate:
    def test_batch_returns_one_result_per_candidate(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        candidates = [
            (
                "Architecture decision: use event sourcing.",
                MemoryDomain.ARCHITECTURE,
                MemoryType.DECISION,
                MemorySource.USER,
            ),
            (
                "Recruiter follow-up pattern: respond within 48 hours.",
                MemoryDomain.RELATIONSHIP,
                MemoryType.FACT,
                MemorySource.USER,
            ),
            (
                "debug: breakpoint() placed in the ORM session code.",
                MemoryDomain.OPERATIONAL,
                MemoryType.FACT,
                MemorySource.SYSTEM,
            ),
        ]
        results = evaluator.batch_evaluate(candidates)
        assert len(results) == 3

    def test_batch_noise_candidate_is_rejected(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        candidates = [
            (
                "debug: added temporary breakpoint() for tracing.",
                MemoryDomain.OPERATIONAL,
                MemoryType.FACT,
                MemorySource.SYSTEM,
            ),
        ]
        results = evaluator.batch_evaluate(candidates)
        assert results[0].retention_decision == RetentionDecision.REJECT

    def test_batch_high_value_candidate_is_retained(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        candidates = [
            (
                "Architecture decision: adopt WAL journal mode. "
                "This is a breaking change to the database interface schema.",
                MemoryDomain.ARCHITECTURE,
                MemoryType.DECISION,
                MemorySource.USER,
            ),
        ]
        results = evaluator.batch_evaluate(candidates)
        assert results[0].retention_decision == RetentionDecision.RETAIN

    def test_batch_empty_returns_empty_list(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        results = evaluator.batch_evaluate([])
        assert results == []


# ---------------------------------------------------------------------------
# Determinism — identical inputs must always produce identical outputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_evaluate_is_deterministic(self, evaluator: SignificanceEvaluator) -> None:
        kwargs: dict = {
            "content": "Architecture decision: adopt WAL mode for SQLite.",
            "domain": MemoryDomain.ARCHITECTURE,
            "memory_type": MemoryType.DECISION,
            "source": MemorySource.USER,
        }
        result_a = evaluator.evaluate(**kwargs)
        result_b = evaluator.evaluate(**kwargs)

        assert result_a.importance_score == result_b.importance_score
        assert result_a.importance_level == result_b.importance_level
        assert result_a.retention_decision == result_b.retention_decision
        assert result_a.score_breakdown == result_b.score_breakdown

    def test_evaluate_existing_is_deterministic(
        self, evaluator: SignificanceEvaluator
    ) -> None:
        entry = _make_entry(
            content="Recruiter Jane prefers async LinkedIn messages over cold email.",
            domain=MemoryDomain.RELATIONSHIP,
            memory_type=MemoryType.FACT,
            source=MemorySource.USER,
        )
        result_a = evaluator.evaluate_existing(entry)
        result_b = evaluator.evaluate_existing(entry)

        assert result_a.importance_score == result_b.importance_score
        assert result_a.retention_decision == result_b.retention_decision

    def test_batch_evaluate_is_deterministic(self, evaluator: SignificanceEvaluator) -> None:
        candidates = [
            (
                "Codebase refactor: split monolithic service into bounded modules.",
                MemoryDomain.CODEBASE,
                MemoryType.DECISION,
                MemorySource.USER,
            ),
            (
                "Operational constraint: API rate limit is 100 requests per minute.",
                MemoryDomain.OPERATIONAL,
                MemoryType.FACT,
                MemorySource.SYSTEM,
            ),
        ]
        results_a = evaluator.batch_evaluate(candidates)
        results_b = evaluator.batch_evaluate(candidates)

        for a, b in zip(results_a, results_b, strict=False):
            assert a.importance_score == b.importance_score
            assert a.retention_decision == b.retention_decision
