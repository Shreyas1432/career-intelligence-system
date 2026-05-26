import pytest

from src.core.ai.service import AIService
from src.modules.matching import (
    AISignal,
    AISponsorshipResult,
    SignalType,
    SponsorshipDetector,
    SponsorshipStatus,
    scan_rules,
)
from tests.fixtures.ai import MockLLMClient


def test_scan_rules_explicit_positive() -> None:
    """
    Test that deterministic rules identify explicit positive sponsorship statements.
    """
    text = "We are looking for a Software Engineer. Visa sponsorship is available for qualified applicants."
    res = scan_rules(text)

    assert res.status == SponsorshipStatus.POSITIVE
    assert res.confidence == 1.0
    assert len(res.signals) == 1
    assert res.signals[0].signal_type == SignalType.SPONSORSHIP_MENTION
    assert res.signals[0].is_positive is True
    assert "visa sponsorship is available" in res.signals[0].matched_text.lower()


def test_scan_rules_explicit_negative() -> None:
    """
    Test that deterministic rules identify explicit negative sponsorship statements.
    """
    text = "Must be authorized to work. We are unable to sponsor visa applications at this time."
    res = scan_rules(text)

    assert res.status == SponsorshipStatus.NEGATIVE
    assert res.confidence == 1.0
    # Should detect both standard work auth (0.8 score, is_positive=False)
    # and explicit negative sponsorship (1.0 score, is_positive=False)
    assert len(res.signals) == 2
    assert any(
        s.signal_type == SignalType.SPONSORSHIP_MENTION and not s.is_positive for s in res.signals
    )
    assert any(s.signal_type == SignalType.WORK_AUTH for s in res.signals)


def test_scan_rules_neutral_work_auth() -> None:
    """
    Test that deterministic rules identify standard work authorization requirements.
    """
    text = "Applicants must be legally authorized to work in the United States."
    res = scan_rules(text)

    assert res.status == SponsorshipStatus.NEUTRAL
    assert res.confidence == 0.8
    assert len(res.signals) >= 1
    assert all(s.signal_type == SignalType.WORK_AUTH for s in res.signals)
    assert all(s.is_positive is False for s in res.signals)


def test_scan_rules_neutral_clues() -> None:
    """
    Test that deterministic rules identify relocation or global team clues.
    """
    text = "Join our global team! We offer excellent benefits including relocation assistance."
    res = scan_rules(text)

    assert res.status == SponsorshipStatus.NEUTRAL
    assert res.confidence == 0.5
    assert len(res.signals) == 2
    assert any(s.signal_type == SignalType.RELOCATION for s in res.signals)
    assert any(s.signal_type == SignalType.GLOBAL_TEAM for s in res.signals)


def test_scan_rules_conflicting() -> None:
    """
    Test that deterministic rules identify conflicting statements.
    """
    text = "H-1B sponsorship is available. Note: Must not require visa sponsorship."
    res = scan_rules(text)

    assert res.status == SponsorshipStatus.NEUTRAL
    assert res.confidence == 0.7
    assert len(res.signals) >= 2


def test_scan_rules_unknown() -> None:
    """
    Test that deterministic rules return unknown when no signals are present.
    """
    text = "Looking for a Python developer with 3 years of experience. Fully remote role."
    res = scan_rules(text)

    assert res.status == SponsorshipStatus.UNKNOWN
    assert res.confidence == 0.0
    assert len(res.signals) == 0


@pytest.mark.asyncio
async def test_detector_rules_bypass_ai(
    mock_ai_service: AIService, mock_llm_client: MockLLMClient
) -> None:
    """
    Test that when rules have high confidence (>= 0.9), AI fallback is not triggered even if use_ai=True.
    """
    detector = SponsorshipDetector(ai_service=mock_ai_service)
    text = "Visa sponsorship is available for this role."

    # Rules will match with confidence 1.0
    res = await detector.detect(text, use_ai=True)

    assert res.status == SponsorshipStatus.POSITIVE
    assert res.confidence == 1.0
    assert len(mock_llm_client.calls) == 0


@pytest.mark.asyncio
async def test_detector_no_ai_fallback_when_disabled(
    mock_ai_service: AIService, mock_llm_client: MockLLMClient
) -> None:
    """
    Test that when rules have low confidence but use_ai=False, AI fallback is not triggered.
    """
    detector = SponsorshipDetector(ai_service=mock_ai_service)
    text = "Standard software engineer description with no visa signals."

    # Rules will match with confidence 0.0
    res = await detector.detect(text, use_ai=False)

    assert res.status == SponsorshipStatus.UNKNOWN
    assert res.confidence == 0.0
    assert len(mock_llm_client.calls) == 0


@pytest.mark.asyncio
async def test_detector_ai_fallback_success(
    mock_ai_service: AIService, mock_llm_client: MockLLMClient
) -> None:
    """
    Test successful AI fallback execution for low-confidence rule outcomes.
    """
    detector = SponsorshipDetector(ai_service=mock_ai_service)
    text = "Standard software engineer description with no visa signals."

    # Set up mock AI response
    ai_result = AISponsorshipResult(
        status="positive",
        confidence=0.95,
        signals=[
            AISignal(
                signal_type="sponsorship_mention",
                matched_text="willing to sponsor the right candidate",
                score=0.95,
                is_positive=True,
            )
        ],
        explanation="AI analysis detected visa sponsorship support.",
    )
    mock_llm_client.add_structured_response(ai_result)

    res = await detector.detect(text, use_ai=True)

    assert res.status == SponsorshipStatus.POSITIVE
    assert res.confidence == 0.95
    assert len(res.signals) == 1
    assert res.signals[0].signal_type == SignalType.SPONSORSHIP_MENTION
    assert res.signals[0].matched_text == "willing to sponsor the right candidate"
    assert res.signals[0].is_positive is True
    assert res.explanation == "AI analysis detected visa sponsorship support."

    assert len(mock_llm_client.calls) == 1
    assert mock_llm_client.calls[0]["type"] == "generate_structured"


@pytest.mark.asyncio
async def test_detector_ai_fallback_resilient_mapping(
    mock_ai_service: AIService, mock_llm_client: MockLLMClient
) -> None:
    """
    Test that AI fallback maps invalid status or signal_type values gracefully.
    """
    detector = SponsorshipDetector(ai_service=mock_ai_service)
    text = "Standard software engineer description."

    ai_result = AISponsorshipResult(
        status="invalid_status",
        confidence=0.6,
        signals=[
            AISignal(
                signal_type="invalid_signal_type",
                matched_text="some text",
                score=0.5,
                is_positive=False,
            )
        ],
        explanation="Fallback check.",
    )
    mock_llm_client.add_structured_response(ai_result)

    res = await detector.detect(text, use_ai=True)

    assert res.status == SponsorshipStatus.UNKNOWN  # Fallback due to invalid status
    assert res.confidence == 0.6
    assert len(res.signals) == 1
    assert res.signals[0].signal_type == SignalType.WORK_AUTH  # Fallback due to invalid signal_type
    assert res.signals[0].matched_text == "some text"


@pytest.mark.asyncio
async def test_detector_ai_fallback_failure_recovery(
    mock_ai_service: AIService, mock_llm_client: MockLLMClient
) -> None:
    """
    Test that detector gracefully recovers to rules result if the AI service raises an error.
    """
    detector = SponsorshipDetector(ai_service=mock_ai_service)
    text = "Join our global team!"

    # Rules will detect global team with confidence 0.5
    rule_res = scan_rules(text)
    assert rule_res.status == SponsorshipStatus.NEUTRAL
    assert rule_res.confidence == 0.5

    # Triggering an exception during structured generation
    # Since structured queue is empty, default validation fails and raise ValueError
    res = await detector.detect(text, use_ai=True)

    # Output should fall back to rules output
    assert res.status == rule_res.status
    assert res.confidence == rule_res.confidence
    assert len(res.signals) == len(rule_res.signals)
    assert len(mock_llm_client.calls) == 3


@pytest.mark.asyncio
async def test_detector_invalid_input() -> None:
    """
    Test that the detector raises TypeError if non-string input is provided.
    """
    detector = SponsorshipDetector()
    with pytest.raises(TypeError, match="job description text must be a string"):
        await detector.detect(12345)  # type: ignore[arg-type]
