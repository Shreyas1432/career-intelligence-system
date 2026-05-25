import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.ai.service import AIService
from src.core.database.models import GovernmentSponsorship

logger = structlog.get_logger("src.modules.matching.sponsorship")


# ------------------------------------------------------------------------------
# Sponsorship Types & Enums
# ------------------------------------------------------------------------------

class SponsorshipStatus(StrEnum):
    """
    Final classification labels for visa sponsorship signal detection.
    """

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class SignalType(StrEnum):
    """
    Categories of clues extracted from job postings.
    """

    WORK_AUTH = "work_auth"
    SPONSORSHIP_MENTION = "sponsorship_mention"
    INTERNATIONAL_WORKFORCE = "international_workforce"
    RELOCATION = "relocation"
    GLOBAL_TEAM = "global_team"


@dataclass(frozen=True, slots=True)
class SponsorshipSignal:
    """
    An isolated clue discovered in a job description.
    """

    signal_type: SignalType
    matched_text: str
    score: float
    is_positive: bool


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """
    Consolidated classification result of the sponsorship detection engine.
    """

    status: SponsorshipStatus
    confidence: float
    signals: list[SponsorshipSignal]
    explanation: str


# ------------------------------------------------------------------------------
# Sponsorship Schemas
# ------------------------------------------------------------------------------

class SponsorshipReasoningMetadata(BaseModel):
    """
    Detailed reasoning component outputs for the combined visa sponsorship evaluation.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    historical_approved_petitions: int = Field(
        ge=0, description="Total historical visa filings approved"
    )
    historical_denied_petitions: int = Field(
        ge=0, description="Total historical visa filings denied"
    )
    extracted_job_status: SponsorshipStatus = Field(
        description="Visa signal extracted from job description"
    )
    extracted_job_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score of the job extraction layer"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Key positive indicators for visa sponsorship"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Key negative indicators or risks for visa sponsorship"
    )
    explanation: str = Field(
        description="Explanatory text summarizing why this score was determined"
    )


class SponsorshipScoringResponse(BaseModel):
    """
    Consolidated visa sponsorship scoring response.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    sponsorship_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Blended probability score for sponsorship friendliness (0-100)",
    )
    sponsorship_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence/reliability weight of the final score (0-1)"
    )
    reasoning: SponsorshipReasoningMetadata = Field(
        description="Breakdown explaining the evaluation components"
    )


# ------------------------------------------------------------------------------
# Sponsorship Rules Engine
# ------------------------------------------------------------------------------

NEGATIVE_SPONSORSHIP_PATTERNS = [
    (re.compile(r"(?i)\bno\s+visa\s+sponsorship\b"), "no visa sponsorship"),
    (
        re.compile(r"(?i)\bnot\s+(?:offering|providing|sponsoring)\s+sponsorship\b"),
        "not offering/providing/sponsoring sponsorship",
    ),
    (re.compile(r"(?i)\bunable\s+to\s+sponsor\b"), "unable to sponsor"),
    (
        re.compile(r"(?i)\bsponsorship\s+(?:is\s+)?not\s+available\b"),
        "sponsorship is not available",
    ),
    (
        re.compile(r"(?i)\bcannot\s+(?:provide|offer|sponsor)\b.*sponsorship"),
        "cannot provide/offer/sponsor sponsorship",
    ),
    (
        re.compile(r"(?i)\bnot\s+eligible\s+for\s+sponsorship\b"),
        "not eligible for sponsorship",
    ),
    (
        re.compile(r"(?i)\bdoes\s+not\s+(?:offer|provide|sponsor)\s+(?:visa\s+)?sponsorship\b"),
        "does not offer/provide/sponsor sponsorship",
    ),
    (
        re.compile(r"(?i)\bmust\s+not\s+require\s+(?:visa\s+)?sponsorship\b"),
        "must not require visa sponsorship",
    ),
    (re.compile(r"(?i)\bno\s+h-?1b\b"), "no h-1b"),
]

POSITIVE_SPONSORSHIP_PATTERNS = [
    (
        re.compile(r"(?i)\bvisa\s+sponsorship\s+(?:is\s+)?available\b"),
        "visa sponsorship is available",
    ),
    (
        re.compile(r"(?i)\bh-?1b\s+sponsorship\s+(?:is\s+)?available\b"),
        "h-1b sponsorship is available",
    ),
    (
        re.compile(r"(?i)\bwe\s+(?:can|will|do)\s+sponsor\b"),
        "we can/will/do sponsor",
    ),
    (re.compile(r"(?i)\bopen\s+to\s+sponsoring\b"), "open to sponsoring"),
    (
        re.compile(r"(?i)\bsponsorship\s+(?:is\s+)?provided\b"),
        "sponsorship is provided",
    ),
    (
        re.compile(r"(?i)\beligible\s+for\s+(?:h-?1b\s+)?sponsorship\b"),
        "eligible for h-1b sponsorship",
    ),
]

WORK_AUTH_PATTERNS = [
    (
        re.compile(r"(?i)\bmust\s+be\s+authorized\s+to\s+work\b"),
        "must be authorized to work",
    ),
    (
        re.compile(r"(?i)\bmust\s+be\s+legally\s+authorized\s+to\s+work\b"),
        "must be legally authorized to work",
    ),
    (
        re.compile(r"(?i)\bproof\s+of\s+work\s+authorization\b"),
        "proof of work authorization",
    ),
    (
        re.compile(r"(?i)\blegally\s+authorized\s+to\s+work\s+in\s+the\b"),
        "legally authorized to work in the",
    ),
]

RELOCATION_PATTERNS = [
    (
        re.compile(r"(?i)\brelocation\s+(?:assistance|support|package|reimbursement)\b"),
        "relocation assistance/support",
    ),
    (re.compile(r"(?i)\bwilling\s+to\s+relocate\b"), "willing to relocate"),
]

GLOBAL_TEAM_PATTERNS = [
    (re.compile(r"(?i)\bglobal\s+team\b"), "global team"),
    (
        re.compile(r"(?i)\bdistributed\s+internationally\b"),
        "distributed internationally",
    ),
    (
        re.compile(r"(?i)\binternational\s+workforce\b"),
        "international workforce",
    ),
    (re.compile(r"(?i)\bmultinational\s+team\b"), "multinational team"),
]


def _scan_pattern_group(
    text: str,
    patterns: list[tuple[re.Pattern[str], str]],
    signal_type: SignalType,
    score: float,
    is_positive: bool,
) -> list[SponsorshipSignal]:
    signals = []
    for pattern, _ in patterns:
        match = pattern.search(text)
        if match:
            signals.append(
                SponsorshipSignal(
                    signal_type=signal_type,
                    matched_text=match.group(0),
                    score=score,
                    is_positive=is_positive,
                )
            )
    return signals


def scan_rules(text: str) -> DetectionResult:
    """
    Deterministic rule-based regex scanner for visa sponsorship detection.
    """
    signals: list[SponsorshipSignal] = []

    # 1. Scan for negative visa sponsorship mentions
    signals.extend(
        _scan_pattern_group(
            text,
            NEGATIVE_SPONSORSHIP_PATTERNS,
            SignalType.SPONSORSHIP_MENTION,
            score=1.0,
            is_positive=False,
        )
    )

    # 2. Scan for positive visa sponsorship mentions
    signals.extend(
        _scan_pattern_group(
            text,
            POSITIVE_SPONSORSHIP_PATTERNS,
            SignalType.SPONSORSHIP_MENTION,
            score=1.0,
            is_positive=True,
        )
    )

    # 3. Scan for neutral work auth requirements
    signals.extend(
        _scan_pattern_group(
            text,
            WORK_AUTH_PATTERNS,
            SignalType.WORK_AUTH,
            score=0.8,
            is_positive=False,
        )
    )

    # 4. Scan for relocation support
    signals.extend(
        _scan_pattern_group(
            text,
            RELOCATION_PATTERNS,
            SignalType.RELOCATION,
            score=0.7,
            is_positive=True,
        )
    )

    # 5. Scan for global team mentions
    signals.extend(
        _scan_pattern_group(
            text,
            GLOBAL_TEAM_PATTERNS,
            SignalType.GLOBAL_TEAM,
            score=0.6,
            is_positive=True,
        )
    )

    status, confidence, explanation = _classify_signals(signals)
    final_signals = signals if status != SponsorshipStatus.UNKNOWN else []

    return DetectionResult(
        status=status,
        confidence=confidence,
        signals=final_signals,
        explanation=explanation,
    )


def _classify_signals(
    signals: list[SponsorshipSignal],
) -> tuple[SponsorshipStatus, float, str]:
    has_pos_sponsorship = any(
        s.signal_type == SignalType.SPONSORSHIP_MENTION and s.is_positive for s in signals
    )
    has_neg_sponsorship = any(
        s.signal_type == SignalType.SPONSORSHIP_MENTION and not s.is_positive for s in signals
    )
    has_work_auth = any(s.signal_type == SignalType.WORK_AUTH for s in signals)

    if has_neg_sponsorship and has_pos_sponsorship:
        return (
            SponsorshipStatus.NEUTRAL,
            0.7,
            "Job description contains conflicting visa sponsorship statements.",
        )
    if has_neg_sponsorship:
        return (
            SponsorshipStatus.NEGATIVE,
            1.0,
            "Explicit visa restrictions or no sponsorship statements detected.",
        )
    if has_pos_sponsorship:
        return (
            SponsorshipStatus.POSITIVE,
            1.0,
            "Explicit visa sponsorship friendly statements detected.",
        )
    if has_work_auth:
        return (
            SponsorshipStatus.NEUTRAL,
            0.8,
            "Standard work authorization statement detected without explicit sponsorship details.",
        )

    has_positive_clues = any(s.is_positive for s in signals)
    if has_positive_clues:
        return (
            SponsorshipStatus.NEUTRAL,
            0.5,
            "International clues detected (e.g. global team/relocation), but visa sponsorship is not explicitly stated.",
        )

    return (
        SponsorshipStatus.UNKNOWN,
        0.0,
        "No sponsorship-friendly, restrictive, or international work clues detected.",
    )


# ------------------------------------------------------------------------------
# Sponsorship Database Persistence
# ------------------------------------------------------------------------------

def normalize_company_name(name: str) -> str:
    """
    Standardize corporate names for exact lookup by stripping suffixes, punctuation, and spaces.
    e.g. "Google LLC" -> "google", "Amazon.com, Inc." -> "amazon com"
    """
    if not name:
        return ""

    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    words = cleaned.split()

    corporate_suffixes = {
        "inc",
        "incorporated",
        "llc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "co",
        "company",
        "group",
        "solutions",
        "technologies",
        "services",
        "systems",
    }

    filtered_words = [w for w in words if w not in corporate_suffixes]
    return " ".join(filtered_words)


class SponsorshipPersistenceService:
    """
    Persistence service coordinating historical government sponsorship records stored in SQLite.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_sponsorship_record(
        self,
        company_name: str,
        fiscal_year: int,
        approved_petitions: int,
        denied_petitions: int,
    ) -> GovernmentSponsorship:
        """
        Create or update a historical government visa sponsorship record.
        """
        normalized = normalize_company_name(company_name)
        total = approved_petitions + denied_petitions

        stmt = select(GovernmentSponsorship).where(
            GovernmentSponsorship.normalized_company_name == normalized,
            GovernmentSponsorship.fiscal_year == fiscal_year,
        )
        record = self.session.scalars(stmt).first()

        if record:
            record.company_name = company_name
            record.approved_petitions = approved_petitions
            record.denied_petitions = denied_petitions
            record.total_petitions = total
        else:
            record = GovernmentSponsorship(
                company_name=company_name,
                normalized_company_name=normalized,
                fiscal_year=fiscal_year,
                approved_petitions=approved_petitions,
                denied_petitions=denied_petitions,
                total_petitions=total,
            )
            self.session.add(record)

        self.session.flush()
        return record

    def get_historical_summary(self, company_name: str) -> dict[str, Any]:
        """
        Query and sum historical sponsorship metrics for a given company.
        """
        normalized = normalize_company_name(company_name)

        stmt = select(GovernmentSponsorship).where(
            GovernmentSponsorship.normalized_company_name == normalized
        )
        records = self.session.scalars(stmt).all()

        if not records:
            return {
                "company_name": company_name,
                "approved": 0,
                "denied": 0,
                "total": 0,
                "has_history": False,
            }

        total_approved = sum(r.approved_petitions for r in records)
        total_denied = sum(r.denied_petitions for r in records)
        total_cases = sum(r.total_petitions for r in records)

        canonical_name = records[0].company_name

        return {
            "company_name": canonical_name,
            "approved": total_approved,
            "denied": total_denied,
            "total": total_cases,
            "has_history": True,
        }


# ------------------------------------------------------------------------------
# Sponsorship Scoring calculations
# ------------------------------------------------------------------------------

def calculate_sponsorship_score(
    history_summary: dict[str, Any],
    extracted_status: SponsorshipStatus,
    extracted_confidence: float,
) -> tuple[float, float, list[str], list[str], str]:
    """
    Deterministically calculate sponsorship friendliness probability (0-100),
    confidence (0-1), and reasoning highlights.
    """
    history_score, history_confidence = _calculate_historical_score_and_confidence(history_summary)

    extract_score, extract_confidence, extract_weight_multiplier = (
        _calculate_extracted_score_and_confidence(extracted_status, extracted_confidence)
    )

    history_weight = history_confidence
    extract_weight = extract_confidence * extract_weight_multiplier
    total_weight = history_weight + extract_weight

    if total_weight > 0.0:
        final_score = (
            history_score * history_weight + extract_score * extract_weight
        ) / total_weight
    else:
        final_score = 50.0

    blended_confidence = 1.0 - (1.0 - history_confidence) * (1.0 - extract_confidence)

    has_history = history_summary.get("has_history", False)
    approved = history_summary.get("approved", 0)
    denied = history_summary.get("denied", 0)

    strengths, gaps = _build_strengths_and_gaps(has_history, approved, denied, extracted_status)

    company_name = history_summary.get("company_name", "the company")

    if final_score >= 80.0:
        fit = "Highly Likely"
    elif final_score >= 60.0:
        fit = "Likely"
    elif final_score >= 40.0:
        fit = "Possible/Neutral"
    else:
        fit = "Unlikely"

    explanation = f"Visa sponsorship for this role at {company_name} is '{fit}' (Score: {round(final_score, 1)}%). "
    if has_history and approved > 0:
        explanation += f"The company has a positive filing history ({approved} approvals). "

    if extracted_status == SponsorshipStatus.NEGATIVE:
        explanation += (
            "However, the job posting explicitly excludes sponsorship for this specific opening."
        )
    elif extracted_status == SponsorshipStatus.POSITIVE:
        explanation += "This is supported by positive language found in the job description."

    return (
        round(final_score, 2),
        round(blended_confidence, 2),
        strengths,
        gaps,
        explanation.strip(),
    )


def _calculate_historical_score_and_confidence(
    history_summary: dict[str, Any],
) -> tuple[float, float]:
    has_history = history_summary.get("has_history", False)
    approved = history_summary.get("approved", 0)
    denied = history_summary.get("denied", 0)
    total = history_summary.get("total", 0)

    if not has_history:
        return 50.0, 0.1

    if approved == 0:
        history_score = 15.0 if denied > 0 else 50.0
    elif approved == 1:
        history_score = 35.0
    elif approved <= 5:
        history_score = 60.0
    elif approved <= 50:
        history_score = 80.0
    else:
        history_score = 90.0

    if total > 10:
        history_confidence = 0.9
    elif total > 0:
        history_confidence = 0.7
    else:
        history_confidence = 0.5

    return history_score, history_confidence


def _calculate_extracted_score_and_confidence(
    extracted_status: SponsorshipStatus,
    extracted_confidence: float,
) -> tuple[float, float, float]:
    if extracted_status == SponsorshipStatus.POSITIVE:
        return 95.0, extracted_confidence, 2.0
    elif extracted_status == SponsorshipStatus.NEGATIVE:
        return 5.0, extracted_confidence, 3.0
    else:
        return 50.0, 0.1, 1.0


def _build_strengths_and_gaps(
    has_history: bool,
    approved: int,
    denied: int,
    extracted_status: SponsorshipStatus,
) -> tuple[list[str], list[str]]:
    strengths = []
    gaps = []

    if has_history:
        if approved > 0:
            strengths.append(
                f"Historical government dataset lists {approved} approved H-1B or green card visa petitions for this company."
            )
        if denied > 0:
            gaps.append(
                f"Historical government dataset lists {denied} denied visa petitions for this company."
            )
    else:
        gaps.append("No historical government visa filing records found for this company name.")

    if extracted_status == SponsorshipStatus.POSITIVE:
        strengths.append("Job description explicitly mentions visa sponsorship is available.")
    elif extracted_status == SponsorshipStatus.NEGATIVE:
        gaps.append("Job description explicitly mentions that visa sponsorship is NOT available.")

    return strengths, gaps


# ------------------------------------------------------------------------------
# Sponsorship Intelligence Engine
# ------------------------------------------------------------------------------

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

        rule_result = scan_rules(text)

        if rule_result.confidence >= 0.9 or not use_ai or not self.ai_service:
            logger.debug(
                "Returning rule-based sponsorship detection result",
                status=rule_result.status,
                confidence=rule_result.confidence,
            )
            return rule_result

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
        history = self.persistence_service.get_historical_summary(company)

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
