from typing import Any

from src.modules.sponsorship.types import SponsorshipStatus


def calculate_sponsorship_score(
    history_summary: dict[str, Any],
    extracted_status: SponsorshipStatus,
    extracted_confidence: float,
) -> tuple[float, float, list[str], list[str], str]:
    """
    Deterministically calculate sponsorship friendliness probability (0-100),
    confidence (0-1), and reasoning highlights.
    """
    # 1. Historical dataset scoring
    history_score, history_confidence = _calculate_historical_score_and_confidence(history_summary)

    # 2. Extracted job signals scoring
    extract_score, extract_confidence, extract_weight_multiplier = (
        _calculate_extracted_score_and_confidence(extracted_status, extracted_confidence)
    )

    # 3. Blending and confidence weighting
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

    # 4. Generate Reasoning strengths and gaps
    has_history = history_summary.get("has_history", False)
    approved = history_summary.get("approved", 0)
    denied = history_summary.get("denied", 0)

    strengths, gaps = _build_strengths_and_gaps(has_history, approved, denied, extracted_status)

    # 5. Verbal explanation
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
        # Neutral or unknown signals are low weight
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
