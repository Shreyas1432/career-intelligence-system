from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import ParseResult


class ATSPlatform(StrEnum):
    """
    Normalized applicant tracking system identifiers.
    """

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    GENERIC_CUSTOM = "generic_custom"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """
    Pre-normalized inputs shared by detection strategies.
    """

    job_url: str
    parsed_url: ParseResult
    host: str
    path: str
    query: str
    html: str | None
    html_text: str
