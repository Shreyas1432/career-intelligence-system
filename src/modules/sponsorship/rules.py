import re

from .types import (
    DetectionResult,
    SignalType,
    SponsorshipSignal,
    SponsorshipStatus,
)

# Pre-compiled regular expressions for clean execution
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
    """
    Scans a group of patterns in the text and returns matching signals.
    """
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

    # Classify decision logic
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
    """
    Analyzes scanned signals to classify sponsorship status, confidence, and explanation.
    """
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

    # Check relocation or global team clues if no direct sponsorship keywords found
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
