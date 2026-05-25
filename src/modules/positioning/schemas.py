from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------------------
# LinkedIn Optimization Schemas
# ------------------------------------------------------------------------------

class ImpactLevel(StrEnum):
    """
    Priority/impact level of profile improvements.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class LinkedInHeadlineOptimization(BaseModel):
    """
    Optimization details for the candidate's LinkedIn headline.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original: str = Field(description="Original headline")
    optimized: str = Field(description="Optimized headline")
    justification: str = Field(
        description="Why this headline was chosen and how it aligns with target roles"
    )


class LinkedInAboutOptimization(BaseModel):
    """
    Optimization details for the candidate's LinkedIn About/Summary section.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original: str = Field(description="Original About section summary")
    optimized: str = Field(description="Optimized About section summary")
    justification: str = Field(
        description="Why this summary was chosen and how it highlights strengths"
    )


class LinkedInExperienceOptimization(BaseModel):
    """
    Optimization details for a specific LinkedIn experience item.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(description="Role title")
    company: str = Field(description="Company name")
    original_description: str = Field(description="Original experience description")
    optimized_description: str = Field(
        description="Optimized experience description incorporating keywords"
    )
    justification: str = Field(
        description="Justification for description updates and keyword insertion"
    )


class LinkedInSkillsOptimization(BaseModel):
    """
    Optimizations for the candidate's featured LinkedIn skills.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    skills_to_add: list[str] = Field(
        default_factory=list, description="Recommended skills to add to profile"
    )
    skills_to_remove_or_deprioritize: list[str] = Field(
        default_factory=list, description="Skills that are less relevant or outdated"
    )
    justification: str = Field(description="Justification for the skill recommendations")


class LinkedInOptimizedProfile(BaseModel):
    """
    Full set of optimized LinkedIn profile sections.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    headline: LinkedInHeadlineOptimization = Field(description="Optimized headline details")
    about: LinkedInAboutOptimization = Field(description="Optimized about section details")
    experiences: list[LinkedInExperienceOptimization] = Field(
        default_factory=list, description="Optimized list of experiences"
    )
    featured_skills: LinkedInSkillsOptimization = Field(description="Optimized skills details")


class RecruiterKeywordAlignment(BaseModel):
    """
    Analysis of profile keyword matches against recruiter search trends.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    high_priority_keywords: list[str] = Field(
        default_factory=list, description="Trending high priority keywords for the target roles"
    )
    matched_keywords: list[str] = Field(
        default_factory=list, description="Keywords already present in the candidate profile"
    )
    missing_keywords: list[str] = Field(
        default_factory=list, description="Keywords that are missing and recommended to add"
    )
    discoverability_index: float = Field(
        ge=0.0, le=100.0, description="Calculated discoverability score out of 100"
    )


class ProfileImprovementSuggestion(BaseModel):
    """
    Actionable suggestion to improve recruiter engagement or structural flow.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    section: str = Field(description="Profile section to improve")
    issue: str = Field(description="Observed issue or gap")
    recommendation: str = Field(description="Actionable fix or advice")
    impact_level: ImpactLevel = Field(description="Expected impact of this improvement")


class LinkedInOptimizationResponse(BaseModel):
    """
    Unified response package of the LinkedIn optimization engine.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    optimized_sections: LinkedInOptimizedProfile = Field(
        description="Structured optimized profile sections"
    )
    keyword_alignment: RecruiterKeywordAlignment = Field(
        description="Recruiter keyword alignment analysis"
    )
    positioning_recommendations: list[str] = Field(
        default_factory=list, description="Broad positioning strategy recommendations"
    )
    improvement_suggestions: list[ProfileImprovementSuggestion] = Field(
        default_factory=list, description="Actionable improvement tips"
    )
    explanation: str = Field(description="Human-readable summary explanation of all decisions")


# ------------------------------------------------------------------------------
# Strategic Positioning Schemas
# ------------------------------------------------------------------------------

class ProjectSchema(BaseModel):
    """
    Structured model for user project details.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(description="Name of the project")
    description: str = Field(description="Details of what was built or solved")
    technologies: list[str] = Field(default_factory=list, description="Technologies or tools used")
    outcome: str = Field(description="Quantitative or qualitative outcome of the project")


class PositioningStatements(BaseModel):
    """
    Concise professional positioning statements.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    headline: str = Field(description="Optimized career headline for the target positioning style")
    elevator_pitch: str = Field(
        description="Brief elevator pitch introducing the candidate's core value"
    )
    focus_areas: list[str] = Field(
        default_factory=list, description="Key professional themes or focus areas"
    )


class DifferentiationAnalysis(BaseModel):
    """
    Objective analysis of candidate's professional differentiators.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    unique_skill_combinations: list[str] = Field(
        default_factory=list,
        description="Unique intersections of technical and domain competencies",
    )
    core_differentiators: list[str] = Field(
        default_factory=list,
        description="Evidence-backed differentiation points derived from metrics",
    )
    market_alignment_score: float = Field(
        ge=0.0, le=100.0, description="Alignment score out of 100 with target market"
    )


class RecruiterFacingSummary(BaseModel):
    """
    recruiter-facing overview and bio sections.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    bio_summary: str = Field(description="recruiter-friendly professional biography")
    value_pillars: list[str] = Field(
        default_factory=list, description="Top skills and methodologies recruiters search for"
    )
    experience_synthesis: str = Field(
        description="Synthesis of career milestones and quantified impacts"
    )


class ValuePropRecommendation(BaseModel):
    """
    Actionable suggestion on how to pitch key candidate value propositions.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value_pillar: str = Field(
        description="Core theme (e.g. Scalability, Process Efficiency, Sourcing)"
    )
    recommendation_text: str = Field(description="Actionable recommendation on what to highlight")
    supporting_evidence: str = Field(
        description="Quantified accomplishment or project metric backing this suggestion"
    )


class StrategicPositioningResponse(BaseModel):
    """
    Unified result package of the strategic positioning analysis.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    positioning_statements: PositioningStatements = Field(
        description="Concise headlines and pitches"
    )
    differentiation: DifferentiationAnalysis = Field(
        description="Strengths and differentiation analysis"
    )
    recruiter_summary: RecruiterFacingSummary = Field(description="recruiter-facing narratives")
    value_prop_recommendations: list[ValuePropRecommendation] = Field(
        default_factory=list, description="Value propositions suggestions"
    )
    explanation: str = Field(description="Human-readable summary explanation of engine decisions")


# ------------------------------------------------------------------------------
# Outreach Context Schemas
# ------------------------------------------------------------------------------

class RecipientProfile(BaseModel):
    """
    Profile information of the contact recipient.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(description="Name of the contact recipient")
    title: str = Field(description="Role title of the recipient")
    company: str = Field(description="Company of the recipient")
    role_type: str = Field(
        description="Type of recipient role (recruiter, engineering_manager, consultant_business, other)"
    )


class RelationshipMetadata(BaseModel):
    """
    Relationship and past contact details.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_degree: str = Field(description="Degree of connection (e.g. 1st, 2nd, cold)")
    past_interactions: list[str] = Field(
        default_factory=list, description="Historical interactions descriptions"
    )


class CommunicationPreferences(BaseModel):
    """
    Candidate preferences for communication style and platform.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    channel: str = Field(description="Preferred platform (e.g. email, linkedin)")
    preferred_tone: str = Field(
        description="Preferred messaging tone (e.g. formal, direct, casual)"
    )


class OpportunityContext(BaseModel):
    """
    Details of the opportunity related to the outreach.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role_title: str = Field(description="Title of the target opportunity role")
    company: str = Field(description="Company offering the target role")
    key_requirements: list[str] = Field(
        default_factory=list, description="Key skills or tools required"
    )


class OutreachInput(BaseModel):
    """
    Unified input representation for outreach generation.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recipient: RecipientProfile = Field(description="Recipient profile details")
    relationship: RelationshipMetadata = Field(description="Relationship metadata")
    preferences: CommunicationPreferences = Field(description="Candidate communication preferences")
    opportunity: OpportunityContext | None = Field(default=None, description="Opportunity context")


class CommunicationDraft(BaseModel):
    """
    Generated outreach message draft.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str | None = Field(default=None, description="Outreach email subject line")
    body: str = Field(description="Core outreach message body text")
    channel: str = Field(description="Platform for this draft (e.g. email, linkedin)")


class OutreachRecommendation(BaseModel):
    """
    Strategic recommendation for sending the message.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    channel_advice: str = Field(description="Platform strategy recommendation")
    best_time_to_send: str = Field(description="Suggested day/time block to send")
    follow_up_cadence_days: int = Field(
        description="Recommended waiting time in days before follow-up"
    )


class OutreachContextResponse(BaseModel):
    """
    Unified response package of the outreach context engine.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    draft: CommunicationDraft = Field(description="Target-oriented outreach draft")
    outreach_recommendations: OutreachRecommendation = Field(description="Outreach recommendations")
    tone_recommendations: list[str] = Field(
        default_factory=list, description="Tone adaptation guidelines"
    )
    follow_up_recommendations: list[str] = Field(
        default_factory=list, description="Cadence and follow-up advice"
    )
    explanation: str = Field(description="Human-readable summary justification of engine decisions")


# ------------------------------------------------------------------------------
# Project Framing Schemas
# ------------------------------------------------------------------------------

class ProjectMetadata(BaseModel):
    """
    Core metadata of the project.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(description="Title of the project")
    role: str = Field(description="Candidate's role in the project")
    description: str = Field(description="General technical description of the project")


class ArchitectureDetails(BaseModel):
    """
    Architectural details of the project.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    design_patterns: list[str] = Field(
        default_factory=list, description="Design patterns used (e.g. Microservices)"
    )
    database_setup: str = Field(description="Database engine and structure details")
    hosting_or_cloud: str = Field(description="Hosting details (e.g. AWS, GCP, On-Premises)")


class ProjectFramingInput(BaseModel):
    """
    Structured input representation for project framing.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    metadata: ProjectMetadata = Field(description="Project title, role, and description")
    architecture: ArchitectureDetails = Field(
        description="Database, cloud, and structural components"
    )
    technologies: list[str] = Field(
        default_factory=list, description="List of tools and languages used"
    )
    business_goals: list[str] = Field(
        default_factory=list, description="Business goals, objectives, and metric targets"
    )


class FramedRecruiterSummary(BaseModel):
    """
    Recruiter-facing simplified overview.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary_text: str = Field(description="Simplified non-technical summary of the project purpose")
    key_outcomes: list[str] = Field(
        default_factory=list, description="Business-oriented outcomes and metric results"
    )


class EnterpriseFraming(BaseModel):
    """
    Framing showing scalability and enterprise relevance.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scalability_narrative: str = Field(
        description="How the project handles scalability and enterprise throughput"
    )
    integration_narrative: str = Field(
        description="Integration with enterprise tools and downstream databases"
    )
    operational_impact: str = Field(
        description="Operational efficiency and financial impact of the project"
    )


class TechnicalExplanation(BaseModel):
    """
    Technical deep dives explaining engineering tradeoffs.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    architectural_decisions: str = Field(
        description="Detailed explanation of system decisions and tradeoffs"
    )
    problem_solving: str = Field(
        description="Engineering reasoning resolving bottlenecks or performance issues"
    )


class PortfolioRecommendation(BaseModel):
    """
    Portfolio formatting and readme suggestions.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    readme_tips: list[str] = Field(
        default_factory=list, description="Actionable tips for improving the GitHub README"
    )
    architecture_visuals_advice: str = Field(
        description="Advice on what system architecture diagrams to include"
    )
    suggested_enhancements: list[str] = Field(
        default_factory=list, description="Potential technical additions to strengthen the project"
    )


class ProjectFramingResponse(BaseModel):
    """
    Unified result package of the project framing engine.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    recruiter_summary: FramedRecruiterSummary = Field(description="Simplified recruiter overview")
    enterprise_framing: EnterpriseFraming = Field(
        description="Enterprise scale and impact narratives"
    )
    technical_explanation: TechnicalExplanation = Field(
        description="Engineering tradeoff explanations"
    )
    portfolio_recommendations: PortfolioRecommendation = Field(
        description="Portfolio presentation advice"
    )
    explanation: str = Field(description="Human-readable justification of framing decisions")
