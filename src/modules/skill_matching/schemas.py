from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.modules.skill_normalization.types import SkillCategory


class MatchType(StrEnum):
    """
    Type of match between a user skill and a job skill.
    """

    EXACT = "exact"
    SEMANTIC = "semantic"


class SkillMatchDetail(BaseModel):
    """
    Detailed match information for a skill matched between user and job.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    matched_skill: str = Field(description="Canonical skill name")
    user_skill: str = Field(description="Original user skill text")
    job_skill: str = Field(description="Original job skill text")
    match_type: MatchType = Field(description="Type of match: exact or semantic")
    similarity: float = Field(
        ge=0.0, le=1.0, description="Cosine similarity or 1.0 for exact match"
    )
    weight: float = Field(
        ge=0.0, description="Weighted importance of the skill in this match context"
    )
    score: float = Field(ge=0.0, description="Calculated score (similarity * weight)")


class MissingSkillDetail(BaseModel):
    """
    Information about required job skills missing from the user profile.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_skill: str = Field(description="Canonical or original required job skill name")
    category: SkillCategory = Field(description="Category classification of the skill")
    weight: float = Field(ge=0.0, description="Weighted importance of the missing skill")


class ScoreBreakdown(BaseModel):
    """
    Deterministic score components for explainable match results.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    exact_match_score: float = Field(ge=0.0, description="Aggregated score from exact matches")
    semantic_match_score: float = Field(
        ge=0.0, description="Aggregated score from semantic matches"
    )
    domain_alignment_bonus: float = Field(
        ge=0.0, description="Bonus score for industry/domain alignment"
    )
    procurement_supply_chain_bonus: float = Field(
        ge=0.0, description="Bonus score for procurement/supply-chain alignment"
    )
    raw_score: float = Field(ge=0.0, description="Total raw score (sum of matched skill scores)")
    total_potential_score: float = Field(
        ge=0.0, description="Total potential score (sum of all job skill weights)"
    )
    normalized_score: float = Field(ge=0.0, le=100.0, description="Percentage score before bonuses")
    final_score: float = Field(ge=0.0, le=100.0, description="Final score capped at 100.0")


class ExplainabilityReport(BaseModel):
    """
    Human-readable explanations, key strengths, gaps, and recommendations.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(description="Overall textual evaluation of the candidate's skill fit")
    strengths: list[str] = Field(
        default_factory=list, description="Key candidate strengths identified"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Key missing skills or capability gaps"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Strategic upskilling/resume recommendations"
    )


class SkillMatchResponse(BaseModel):
    """
    Unified result returned by the Skill Matching Engine.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    final_score: float = Field(ge=0.0, le=100.0, description="Overall matching score")
    matched_skills: list[SkillMatchDetail] = Field(
        default_factory=list, description="List of matched skills"
    )
    missing_skills: list[MissingSkillDetail] = Field(
        default_factory=list, description="List of missing required skills"
    )
    score_breakdown: ScoreBreakdown = Field(description="Detailed components of the final score")
    explanation: ExplainabilityReport = Field(
        description="Explainability and actionable feedback report"
    )
