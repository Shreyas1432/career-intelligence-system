from dataclasses import dataclass
from enum import StrEnum


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
