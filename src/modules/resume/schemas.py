
from pydantic import BaseModel, ConfigDict, Field


class ATSMatchMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    keyword_alignment_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    keyword_alignment_ratio: float

class ATSOptimization(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    target_keywords: list[str]
    missing_keywords_to_add: list[str]
    resume_section_recommendations: dict[str, list[str]] = Field(default_factory=dict)

class EmphasizedSkill(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    skill_name: str
    importance: str
    user_possesses: bool
    rationale: str

class PositioningRecommendation(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    suggested_headline: str
    recommended_focus_areas: list[str]
    positioning_pitch: str

class PrioritizedExperience(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    title: str
    company: str
    priority_score: float
    priority_band: str
    justification: str

class RenderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    output_path: str
    format: str
    html_content: str

class TailoredExperienceItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    title: str
    company: str
    start_date: str
    end_date: str | None = None
    description: str
    bullets: list[str]

class TailoredResumeStructure(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    full_name: str
    email: str
    suggested_headline: str
    professional_summary: str
    experiences: list[TailoredExperienceItem]
    skills: list[str]
    phone: str | None = None
    github: str | None = None
    linkedin: str | None = None

class TailoredResumeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    tailored_resume: TailoredResumeStructure
    ats_metadata: ATSMatchMetadata
    missing_skill_suggestions: list[str]

class TailoringStrategyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    overall_alignment_summary: str
    resume_tailoring_strategy: str
    prioritized_experiences: list[PrioritizedExperience]
    emphasized_skills: list[EmphasizedSkill]
    positioning_recommendations: PositioningRecommendation
    ats_optimization: ATSOptimization
    explanation: str

class TemplateStyleConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    font_family: str = "Inter, sans-serif"
    primary_color: str = "#1A365D"
    secondary_color: str = "#4A5568"
    text_color: str = "#2D3748"
    margin_top: str = "0.75in"
    margin_bottom: str = "0.75in"
    margin_left: str = "0.75in"
    margin_right: str = "0.75in"
    section_order: list[str] = Field(default_factory=lambda: ["summary", "experience", "skills"])
