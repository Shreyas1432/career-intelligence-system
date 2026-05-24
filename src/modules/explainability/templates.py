from typing import Any


def render_score_composition(weights: Any, factors: Any, overall_score: float) -> str:
    """
    Format a math-style textual representation of the weighted overall score.
    """

    def _get(obj: Any, key: str) -> float:
        if isinstance(obj, dict):
            return float(obj.get(key, 0.0))
        return float(getattr(obj, key, 0.0))

    w_skill = _get(weights, "skill_matching")
    w_domain = _get(weights, "domain_alignment")
    w_spon = _get(weights, "sponsorship_probability")
    w_exp = _get(weights, "experience_relevance")
    w_ent = _get(weights, "enterprise_alignment")

    s_skill = _get(factors, "skill_matching")
    s_domain = _get(factors, "domain_alignment")
    s_spon = _get(factors, "sponsorship_probability")
    s_exp = _get(factors, "experience_relevance")
    s_ent = _get(factors, "enterprise_alignment")

    explanation = (
        f"The overall score of {overall_score}% is composed of five weighted factors:\n"
        f"- Skill Matching: {s_skill}% score at {w_skill:.0%} weight\n"
        f"- Domain Alignment: {s_domain}% score at {w_domain:.0%} weight\n"
        f"- Visa Sponsorship: {s_spon}% score at {w_spon:.0%} weight\n"
        f"- Experience Relevance: {s_exp}% score at {w_exp:.0%} weight\n"
        f"- Enterprise Alignment: {s_ent}% score at {w_ent:.0%} weight\n"
        f"Calculation: ({s_skill} * {w_skill}) + ({s_domain} * {w_domain}) + "
        f"({s_spon} * {w_spon}) + ({s_exp} * {w_exp}) + ({s_ent} * {w_ent}) = {overall_score}%"
    )
    return explanation


def render_recruiter_summary(
    fit_label: str,
    overall_score: float,
    company: str,
    title: str,
    highlights: list[str],
    concerns: list[str],
    sponsorship_text: str,
) -> str:
    """
    Synthesize recruiter-style narrative reviewing candidate compatibility.
    """
    highlight_text = " ".join(highlights) if highlights else "No major highlights identified."
    concern_text = " ".join(concerns) if concerns else "No significant concerns identified."
    sponsorship_part = (
        sponsorship_text if sponsorship_text else "Sponsorship details not evaluated."
    )

    narrative = (
        f"Recruiter Assessment: Candidate presents as a '{fit_label}' (Overall Alignment: {overall_score}%) "
        f"for the '{title}' role at '{company}'. "
        f"{highlight_text} "
        f"{concern_text} "
        f"{sponsorship_part}"
    )
    return narrative
