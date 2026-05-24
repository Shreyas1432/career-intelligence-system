from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RecommendationCategory(StrEnum):
    """
    Actionable recommendation categories.
    """

    STRONG_APPLY = "strong_apply"
    APPLY = "apply"
    WEAK_APPLY = "weak_apply"
    SKIP = "skip"


class RankingWeights(BaseModel):
    """
    Configurable relative factor weights for opportunity ranking.
    """

    model_config = ConfigDict(extra="forbid")

    skill_matching: float = Field(
        default=0.30, ge=0.0, description="Weight of core skill matching (0-1)"
    )
    domain_alignment: float = Field(
        default=0.20, ge=0.0, description="Weight of domain taxonomy alignment (0-1)"
    )
    sponsorship_probability: float = Field(
        default=0.20, ge=0.0, description="Weight of visa sponsorship signals (0-1)"
    )
    experience_relevance: float = Field(
        default=0.15, ge=0.0, description="Weight of experience years/seniority alignment (0-1)"
    )
    enterprise_alignment: float = Field(
        default=0.15, ge=0.0, description="Weight of target roles and industry preferences (0-1)"
    )

    @model_validator(mode="after")
    def normalize_or_validate_weights(self) -> Self:
        """
        Verify that total weights are non-zero and optionally normalize them.
        """
        total = (
            self.skill_matching
            + self.domain_alignment
            + self.sponsorship_probability
            + self.experience_relevance
            + self.enterprise_alignment
        )
        if total <= 0.0:
            raise ValueError("Sum of weights must be greater than zero")

        # We normalize weights to sum exactly to 1.0
        self.skill_matching = round(self.skill_matching / total, 4)
        self.domain_alignment = round(self.domain_alignment / total, 4)
        self.sponsorship_probability = round(self.sponsorship_probability / total, 4)
        self.experience_relevance = round(self.experience_relevance / total, 4)
        self.enterprise_alignment = round(self.enterprise_alignment / total, 4)
        return self


class FactorScores(BaseModel):
    """
    Component scores out of 100 for each evaluated opportunity factor.
    """

    model_config = ConfigDict(extra="forbid")

    skill_matching: float = Field(ge=0.0, le=100.0)
    domain_alignment: float = Field(ge=0.0, le=100.0)
    sponsorship_probability: float = Field(ge=0.0, le=100.0)
    experience_relevance: float = Field(ge=0.0, le=100.0)
    enterprise_alignment: float = Field(ge=0.0, le=100.0)


class RankingReasoning(BaseModel):
    """
    Explainability indicators for opportunity ranking results.
    """

    model_config = ConfigDict(extra="forbid")

    strengths: list[str] = Field(
        default_factory=list, description="Top positive evaluation indicators"
    )
    gaps: list[str] = Field(default_factory=list, description="Critical deficiencies or risks")
    explanation: str = Field(
        description="Paragraph explanation justifying the final recommendation"
    )


class OpportunityRankingResponse(BaseModel):
    """
    Unified ranking response.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    overall_score: float = Field(ge=0.0, le=100.0, description="Blended overall score out of 100")
    recommendation: RecommendationCategory = Field(description="Actionable category mapping")
    factors: FactorScores = Field(description="Calculated factor score breakdown")
    weights: RankingWeights = Field(description="Normalized weights applied")
    reasoning: RankingReasoning = Field(description="Explainability and feedback metadata")
