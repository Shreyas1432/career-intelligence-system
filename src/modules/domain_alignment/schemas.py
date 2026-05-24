from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DomainCategory(StrEnum):
    """
    Key target domains for career intelligence domain alignment.
    """

    ENTERPRISE_SYSTEMS = "enterprise_systems"
    PROCUREMENT = "procurement"
    SUPPLY_CHAIN = "supply_chain"
    AI_ANALYTICS = "ai_analytics"


class DomainScoreDetails(BaseModel):
    """
    Score components and matched keywords for a single domain.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: float = Field(ge=0.0, le=100.0, description="Overall blended score for this domain")
    rule_score: float = Field(ge=0.0, le=100.0, description="Rule-based keyword match score")
    semantic_score: float = Field(ge=0.0, le=100.0, description="Semantic similarity match score")
    matched_keywords: list[str] = Field(
        default_factory=list, description="Keywords matched for this domain"
    )


class ReasoningMetadata(BaseModel):
    """
    Explainability report details for domain alignment.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    semantic_similarity: float = Field(
        ge=0.0, le=1.0, description="Raw cosine similarity between positioning and job details"
    )
    matched_keywords: list[str] = Field(
        default_factory=list, description="All keywords matched across all domains"
    )
    strengths: list[str] = Field(default_factory=list, description="Identified domain strengths")
    gaps: list[str] = Field(default_factory=list, description="Identified gaps in domain alignment")
    explanation: str = Field(description="Paragraph explanation justifying the alignment score")


class DomainAlignmentResponse(BaseModel):
    """
    Unified response representing the final domain alignment score and reasoning metadata.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    final_score: float = Field(
        ge=0.0, le=100.0, description="Aggregated overall domain alignment score"
    )
    domain_breakdown: dict[DomainCategory, DomainScoreDetails] = Field(
        description="Detailed score breakdowns for each taxonomy domain"
    )
    reasoning: ReasoningMetadata = Field(description="Explainability and feedback metadata")
