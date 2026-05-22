from .engine import SponsorshipDetector
from .rules import scan_rules
from .types import (
    DetectionResult,
    SignalType,
    SponsorshipSignal,
    SponsorshipStatus,
)

__all__ = [
    "DetectionResult",
    "SignalType",
    "SponsorshipDetector",
    "SponsorshipSignal",
    "SponsorshipStatus",
    "scan_rules",
]
