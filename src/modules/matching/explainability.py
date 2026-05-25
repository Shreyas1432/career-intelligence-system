from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------------------
# Explainability Schemas
# ------------------------------------------------------------------------------

class ExplainabilityLayerResponse(BaseModel):
    """
    Unified explainability report containing recruiter summaries,
    composition math explanations, and tailored improvement recommendations.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    recruiter_summary: str = Field(
        description="Recruiter-style text narrative reviewing the candidate fit"
    )
    score_composition_explanation: str = Field(
        description="Math explanation of how weighted factor scores compose the final result"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Consolidated list of candidate strengths"
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Consolidated list of gaps or areas of improvement"
    )
    actionable_insights: list[str] = Field(
        default_factory=list, description="Feasibility, alignment, and strategic insights"
    )
    improvement_recommendations: list[str] = Field(
        default_factory=list, description="Concrete resume or upskilling recommendations"
    )


# ------------------------------------------------------------------------------
# Explainability Templates
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Explainability Service
# ------------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ExplainabilityService:
    """
    Consolidated explanation service for career intelligence.
    Extracts, blends, and explains skill match, domain alignment, visa sponsorship,
    and opportunity ranking scores into recruiter-style formats.
    """

    def generate_explanation(
        self,
        skill_match: Any,
        domain_align: Any,
        sponsorship: Any,
        ranking: Any,
        company: str,
        title: str,
    ) -> ExplainabilityLayerResponse:
        """
        Orchestrate the compilation and rendering of the consolidated explainability report.
        """
        overall_score = float(_get(ranking, "overall_score", 50.0))
        weights = _get(ranking, "weights", {})
        factors = _get(ranking, "factors", {})
        recommendation = str(_get(ranking, "recommendation", "skip"))

        ranking_reasoning = _get(ranking, "reasoning", {})
        matched_skills = _get(skill_match, "matched_skills", [])
        missing_skills = _get(skill_match, "missing_skills", [])

        strengths = self._extract_strengths(ranking_reasoning, matched_skills)
        weaknesses = self._extract_weaknesses(ranking_reasoning, missing_skills)

        sponsorship_reasoning = _get(sponsorship, "reasoning", {})
        spon_score = float(_get(sponsorship, "sponsorship_score", 50.0))
        approved = int(_get(sponsorship_reasoning, "historical_approved_petitions", 0))
        domain_reasoning = _get(domain_align, "reasoning", {})
        all_matched_kws = _get(domain_reasoning, "matched_keywords", [])

        actionable_insights = self._build_actionable_insights(
            spon_score, approved, all_matched_kws, company
        )

        improvement_recommendations = self._build_improvement_recommendations(
            missing_skills, matched_skills
        )

        fit_label = recommendation.replace("_", " ").title()
        spon_explanation = str(_get(sponsorship_reasoning, "explanation", ""))

        recruiter_summary = render_recruiter_summary(
            fit_label=fit_label,
            overall_score=overall_score,
            company=company,
            title=title,
            highlights=strengths[:2],
            concerns=weaknesses[:2],
            sponsorship_text=spon_explanation,
        )

        score_explanation = render_score_composition(weights, factors, overall_score)

        return ExplainabilityLayerResponse(
            recruiter_summary=recruiter_summary,
            score_composition_explanation=score_explanation,
            strengths=list(dict.fromkeys(strengths)),
            weaknesses=list(dict.fromkeys(weaknesses)),
            actionable_insights=list(dict.fromkeys(actionable_insights)),
            improvement_recommendations=list(dict.fromkeys(improvement_recommendations)),
        )

    def _extract_strengths(self, ranking_reasoning: Any, matched_skills: list[Any]) -> list[str]:
        strengths = []
        strengths.extend(_get(ranking_reasoning, "strengths", []))

        for md in matched_skills:
            m_type = _get(md, "match_type", "")
            matched_skill = _get(md, "matched_skill", "")
            if m_type == "exact":
                strengths.append(f"Technical fit: Exact match in '{matched_skill}'.")
            elif m_type == "semantic":
                user_skill = _get(md, "user_skill", "")
                job_skill = _get(md, "job_skill", "")
                strengths.append(
                    f"Transferable skill: '{user_skill}' covers required '{job_skill}'."
                )
        return strengths

    def _extract_weaknesses(self, ranking_reasoning: Any, missing_skills: list[Any]) -> list[str]:
        weaknesses = []
        weaknesses.extend(_get(ranking_reasoning, "gaps", []))

        for ms in missing_skills:
            job_skill = _get(ms, "job_skill", "")
            category = _get(ms, "category", "")
            weaknesses.append(f"Skill gap: Missing required '{job_skill}' (Category: {category}).")
        return weaknesses

    def _build_actionable_insights(
        self,
        spon_score: float,
        approved: int,
        all_matched_kws: list[str],
        company: str,
    ) -> list[str]:
        actionable_insights = []

        if spon_score >= 80.0:
            actionable_insights.append(
                f"Highly favorable sponsorship outlook at {company} (filing history lists {approved} approvals)."
            )
        elif spon_score < 40.0:
            actionable_insights.append(
                "Sponsorship probability is low. Prepare for potential work authorization discussions early."
            )
        else:
            actionable_insights.append("Sponsorship probability is neutral or unconfirmed.")

        if all_matched_kws:
            actionable_insights.append(
                f"Your positioning overlaps with job domains using key terms: {', '.join(all_matched_kws)}."
            )
        return actionable_insights

    def _build_improvement_recommendations(
        self, missing_skills: list[Any], matched_skills: list[Any]
    ) -> list[str]:
        improvement_recommendations = []

        for ms in missing_skills[:2]:
            job_skill = _get(ms, "job_skill", "")
            improvement_recommendations.append(
                f"Upskill in '{job_skill}' to address a critical required skill."
            )

        for md in matched_skills:
            m_type = _get(md, "match_type", "")
            if m_type == "semantic":
                user_skill = _get(md, "user_skill", "")
                job_skill = _get(md, "job_skill", "")
                improvement_recommendations.append(
                    f"Explicitly mention your experience with '{job_skill}' (currently matching via '{user_skill}')."
                )
                break

        if not improvement_recommendations:
            improvement_recommendations.append(
                "Your profile aligns well. Consider tailoring your cover letter to highlight matching enterprise domains."
            )
        return improvement_recommendations
