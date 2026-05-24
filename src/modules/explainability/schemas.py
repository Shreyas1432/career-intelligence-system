from pydantic import BaseModel, ConfigDict, Field


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
