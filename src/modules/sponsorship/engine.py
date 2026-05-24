from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.core.ai.service import AIService
from src.modules.sponsorship.persistence import SponsorshipPersistenceService
from src.modules.sponsorship.schemas import SponsorshipReasoningMetadata, SponsorshipScoringResponse
from src.modules.sponsorship.scoring import calculate_sponsorship_score

from .rules import scan_rules
from .types import (
    DetectionResult,
    SignalType,
    SponsorshipSignal,
    SponsorshipStatus,
)

logger = structlog.get_logger("src.modules.sponsorship.engine")


class AISignal(BaseModel):
    """
    Pydantic structure for individual signals extracted by LLM.
    """

    signal_type: str = Field(
        description="One of: work_auth, sponsorship_mention, international_workforce, relocation, global_team"
    )
    matched_text: str = Field(description="The exact text snippet that matches the signal")
    score: float = Field(
        description="Confidence score for this specific signal, between 0.0 and 1.0"
    )
    is_positive: bool = Field(
        description="True if the signal is positive/supportive, False if negative/restrictive"
    )


class AISponsorshipResult(BaseModel):
    """
    Pydantic structure for total LLM sponsorship analysis.
    """

    status: str = Field(description="One of: positive, neutral, negative, unknown")
    confidence: float = Field(description="Overall confidence score between 0.0 and 1.0")
    signals: list[AISignal] = Field(description="List of detected signals")
    explanation: str = Field(description="Detailed explanation of the classification decision")


class SponsorshipDetector:
    """
    Hybrid visa sponsorship signal detector. Runs deterministic rules
    first for fast, O(1) matching. Falls back to AI if results are unknown
    and AI is enabled.
    """

    def __init__(self, ai_service: AIService | None = None) -> None:
        self.ai_service = ai_service

    async def detect(self, text: str, use_ai: bool = False) -> DetectionResult:
        """
        Analyze job description text for visa sponsorship signals.
        """
        if not isinstance(text, str):
            raise TypeError("job description text must be a string")

        # 1. Run deterministic rules engine first
        rule_result = scan_rules(text)

        # 2. Return immediately if rules found a high confidence match (1.0)
        # or if AI fallback is disabled / no AI service is injected
        if rule_result.confidence >= 0.9 or not use_ai or not self.ai_service:
            logger.debug(
                "Returning rule-based sponsorship detection result",
                status=rule_result.status,
                confidence=rule_result.confidence,
            )
            return rule_result

        # 3. Fall back to AI layer for deep semantic understanding of low confidence/unknown rules outcomes
        logger.info(
            "Invoking AI enhancement layer for sponsorship signal analysis",
            rules_status=rule_result.status,
            rules_confidence=rule_result.confidence,
        )

        try:
            ai_res = await self.ai_service.generate_structured_from_template(
                template_path="sponsorship/detect.md",
                context={"job_description": text},
                response_model=AISponsorshipResult,
            )

            # Map the Pydantic AI result to our Python dataclasses
            mapped_signals = []
            for sig in ai_res.signals:
                try:
                    sig_type = SignalType(sig.signal_type)
                except ValueError:
                    sig_type = SignalType.WORK_AUTH

                mapped_signals.append(
                    SponsorshipSignal(
                        signal_type=sig_type,
                        matched_text=sig.matched_text,
                        score=sig.score,
                        is_positive=sig.is_positive,
                    )
                )

            try:
                status = SponsorshipStatus(ai_res.status)
            except ValueError:
                status = SponsorshipStatus.UNKNOWN

            return DetectionResult(
                status=status,
                confidence=ai_res.confidence,
                signals=mapped_signals,
                explanation=ai_res.explanation,
            )

        except Exception as exc:
            logger.error(
                "Sponsorship detection AI enhancement failed, falling back to rule result",
                error=str(exc),
            )
            return rule_result


class SponsorshipScoringEngine:
    """
    Combined visa sponsorship intelligence scoring engine.
    Integrates historical government data with real-time job posting signals.
    """

    def __init__(
        self,
        persistence_service: SponsorshipPersistenceService,
        detector: SponsorshipDetector | None = None,
    ) -> None:
        self.persistence_service = persistence_service
        self.detector = detector or SponsorshipDetector()

    async def evaluate_sponsorship(
        self,
        company: str,
        job_description: str | None = None,
        extracted_signals: Any = None,
        use_ai: bool = False,
    ) -> SponsorshipScoringResponse:
        """
        Evaluate and score a company's visa sponsorship likelihood.
        """
        # 1. Fetch historical record summary
        history = self.persistence_service.get_historical_summary(company)

        # 2. Get real-time job posting signals
        extracted_status = SponsorshipStatus.UNKNOWN
        extracted_confidence = 0.0

        if extracted_signals is not None:
            if isinstance(extracted_signals, DetectionResult):
                extracted_status = extracted_signals.status
                extracted_confidence = extracted_signals.confidence
            elif isinstance(extracted_signals, dict):
                status_raw = extracted_signals.get("status", "unknown")
                try:
                    extracted_status = SponsorshipStatus(status_raw)
                except ValueError:
                    extracted_status = SponsorshipStatus.UNKNOWN
                extracted_confidence = extracted_signals.get("confidence", 0.0)
        elif job_description:
            det_res = await self.detector.detect(job_description, use_ai=use_ai)
            extracted_status = det_res.status
            extracted_confidence = det_res.confidence

        # 3. Perform blended scoring calculations
        score, confidence, strengths, gaps, explanation = calculate_sponsorship_score(
            history_summary=history,
            extracted_status=extracted_status,
            extracted_confidence=extracted_confidence,
        )

        reasoning = SponsorshipReasoningMetadata(
            historical_approved_petitions=history.get("approved", 0),
            historical_denied_petitions=history.get("denied", 0),
            extracted_job_status=extracted_status,
            extracted_job_confidence=extracted_confidence,
            strengths=strengths,
            gaps=gaps,
            explanation=explanation,
        )

        return SponsorshipScoringResponse(
            sponsorship_score=score,
            sponsorship_confidence=confidence,
            reasoning=reasoning,
        )
