from pydantic import BaseModel, ConfigDict, Field

from src.modules.sponsorship.types import SponsorshipStatus


class SponsorshipReasoningMetadata(BaseModel):
    """
    Detailed reasoning component outputs for the combined visa sponsorship evaluation.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    historical_approved_petitions: int = Field(
        ge=0, description="Total historical visa filings approved"
    )
    historical_denied_petitions: int = Field(
        ge=0, description="Total historical visa filings denied"
    )
    extracted_job_status: SponsorshipStatus = Field(
        description="Visa signal extracted from job description"
    )
    extracted_job_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score of the job extraction layer"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Key positive indicators for visa sponsorship"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Key negative indicators or risks for visa sponsorship"
    )
    explanation: str = Field(
        description="Explanatory text summarizing why this score was determined"
    )


class SponsorshipScoringResponse(BaseModel):
    """
    Consolidated visa sponsorship scoring response.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    sponsorship_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Blended probability score for sponsorship friendliness (0-100)",
    )
    sponsorship_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence/reliability weight of the final score (0-1)"
    )
    reasoning: SponsorshipReasoningMetadata = Field(
        description="Breakdown explaining the evaluation components"
    )
