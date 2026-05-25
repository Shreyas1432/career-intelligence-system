import re
from typing import Any

from src.modules.positioning.schemas import (
    EnterpriseFraming,
    FramedRecruiterSummary,
    PortfolioRecommendation,
    ProjectFramingResponse,
    TechnicalExplanation,
)

# Replaces generic social-media or marketing buzzwords with clear professional counterparts
BUZZWORDS_MAP = {
    r"\bdisruptive\b": "Strategic",
    r"\bdisruptor\b": "Innovator",
    r"\bvisionary\b": "Strategist",
    r"\bguru\b": "Expert",
    r"\bninja\b": "Specialist",
    r"\brockstar\b": "Professional",
    r"\bevangelist\b": "Advocate",
    r"\btransformative\b": "Strategic",
    r"\bcutting-edge\b": "Advanced",
    r"\bworld-class\b": "High-performing",
    r"\brevolutionary\b": "Innovative",
    r"\bgame-changer\b": "Catalyst",
    r"\bgame-changing\b": "Innovative",
    r"\bpassionate\b": "Dedicated",
}


# ------------------------------------------------------------------------------
# Project Narrative Generator
# ------------------------------------------------------------------------------

class ProjectNarrativeGenerator:
    """
    Translates technical details into business-oriented narratives and structural engineering tradeoffs.
    Enforces anti-buzzword constraints.
    """

    def clean_buzzwords(self, text: str) -> str:
        """
        Strips subjective marketing fluff and replaces it with professional counterparts.
        """
        if not text:
            return ""

        cleaned = text
        for pattern, replacement in BUZZWORDS_MAP.items():

            def replace_match(match: re.Match[str], repl: str = replacement) -> str:
                val = match.group(0)
                if val.istitle():
                    return repl.title()
                if val.isupper():
                    return repl.upper()
                return repl

            cleaned = re.sub(pattern, replace_match, cleaned, flags=re.IGNORECASE)

        return cleaned

    def generate_recruiter_summary(
        self, metadata: dict[str, Any], business_goals: list[str]
    ) -> FramedRecruiterSummary:
        """
        Generates simplified recruiter summaries highlighting business objectives and outcomes.
        """
        title = metadata.get("title", "Project")
        role = metadata.get("role", "Developer")
        desc = metadata.get("description", "") or ""

        # Convert simple database scripting descriptors to high-value terms
        business_desc = desc
        business_desc = re.sub(
            r"\bwriting scripts\b",
            "orchestrating automation workflows",
            business_desc,
            flags=re.IGNORECASE,
        )
        business_desc = re.sub(
            r"\bdatabase queries\b", "data integrity systems", business_desc, flags=re.IGNORECASE
        )

        summary_text = (
            f"Under the project '{title}', served as '{role}' to design and build system components. "
            f"Focus was on {self.clean_buzzwords(business_desc)}"
        )

        outcomes = []
        for goal in business_goals:
            outcomes.append(
                f"Successfully delivered target objective: {self.clean_buzzwords(goal)}"
            )

        # Fallback if no business goals are listed
        if not outcomes:
            outcomes.append(
                "Successfully optimized system components to improve overall operational stability."
            )

        return FramedRecruiterSummary(
            summary_text=summary_text,
            key_outcomes=outcomes,
        )

    def generate_enterprise_framing(
        self,
        metadata: dict[str, Any],
        architecture: dict[str, Any],
        technologies: list[str],
        business_goals: list[str],
    ) -> EnterpriseFraming:
        """
        Frames the project to highlight enterprise scalability, integration, and operational impact.
        """
        title = metadata.get("title", "Project")
        patterns = architecture.get("design_patterns", []) or ["Modular Design"]
        patterns_str = ", ".join(patterns)
        hosting = architecture.get("hosting_or_cloud", "Cloud Platforms")

        tech_str = ", ".join(technologies[:3]) if technologies else "modern tech stacks"

        scalability = (
            f"Designed the project '{title}' using '{patterns_str}' patterns on '{hosting}' to ensure "
            f"high availability, high throughput, and linear scalability under peak user demands."
        )

        db = architecture.get("database_setup", "databases")
        integration = (
            f"Configured integration layouts connecting {tech_str} to database structures ({db}). "
            f"Ensured data consistency across downstream analytical tables and enterprise applications."
        )

        goals_str = (
            " and ".join(business_goals[:2])
            if business_goals
            else "optimizing system runtime metrics"
        )
        impact = (
            f"Delivered key operational metrics by aligning engineering architecture with business goals, "
            f"specifically targeting {goals_str}."
        )

        return EnterpriseFraming(
            scalability_narrative=self.clean_buzzwords(scalability),
            integration_narrative=self.clean_buzzwords(integration),
            operational_impact=self.clean_buzzwords(impact),
        )

    def generate_technical_explanation(
        self,
        _metadata: dict[str, Any],
        architecture: dict[str, Any],
        technologies: list[str],
    ) -> TechnicalExplanation:
        """
        Generates deep-dive engineering tradeoff and problem-solving sections.
        """
        patterns = architecture.get("design_patterns", []) or ["Modular boundaries"]
        tech_str = ", ".join(technologies[:3]) if technologies else "selected frameworks"
        db = architecture.get("database_setup", "databases")

        decisions = (
            f"Adopted '{', '.join(patterns)}' architectures using {tech_str}. This design separates "
            f"core logic boundaries, reduces database connection overhead, and simplifies testing. "
            f"This tradeoff balances operational maintenance overhead against technical decoupling advantages."
        )

        solving = (
            f"Addressed performance bottlenecks associated with '{db}' setups. "
            f"Implemented caching boundaries and query indexing, resolving data latency bottlenecks "
            f"to meet performance objectives."
        )

        return TechnicalExplanation(
            architectural_decisions=self.clean_buzzwords(decisions),
            problem_solving=self.clean_buzzwords(solving),
        )


# ------------------------------------------------------------------------------
# Project Recommendation Layer
# ------------------------------------------------------------------------------

class ProjectRecommendationLayer:
    """
    Generates GitHub README formatting tips, system diagram suggestions,
    and potential technical additions to strengthen the project's portfolio presence.
    """

    def generate_portfolio_recommendations(
        self,
        technologies: list[str],
        architecture: dict[str, Any],
        business_goals: list[str],
    ) -> PortfolioRecommendation:
        """
        Builds formatting and expansion advice.
        """
        hosting = architecture.get("hosting_or_cloud", "Cloud Platforms")
        db = architecture.get("database_setup", "databases")

        # 1. Action-oriented README tips
        tips = [
            f"Include a System Architecture diagram showing how {hosting} components connect to {db}.",
            f"Add a structured 'Tech Stack' table highlighting core technologies: {', '.join(technologies[:4])}.",
        ]

        if business_goals:
            tips.append(
                f"Create a dedicated 'Business Impact' section featuring your main outcomes: {', '.join(business_goals[:2])}."
            )
        else:
            tips.append(
                "Add a section highlighting quantified performance and stability improvements."
            )

        # 2. Visual diagram suggestion
        visuals_advice = (
            f"Add a container diagram (C4 Model Level 2) visualizing client connections, "
            f"backend API services, and the '{db}' database layer hosted on '{hosting}'."
        )

        # 3. Technical enhancements to strengthen the project representation
        enhancements = [
            "Implement structured JSON logging and observability tracing (e.g. OpenTelemetry) for enterprise monitoring.",
            "Introduce a decoupled messaging queue (e.g. RabbitMQ, AWS SQS) to process background tasks asynchronously.",
            "Add automated integration tests checking database transaction rollbacks and API schema boundaries.",
        ]

        return PortfolioRecommendation(
            readme_tips=tips,
            architecture_visuals_advice=visuals_advice,
            suggested_enhancements=enhancements,
        )


# ------------------------------------------------------------------------------
# Project Framing Engine
# ------------------------------------------------------------------------------

def _get_val(obj: Any, field: str, default: Any = None) -> Any:
    """
    Safely retrieves a field value from a dictionary, Pydantic model, or arbitrary object.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


class ProjectFramingEngine:
    """
    Core engine orchestrating technical project framing analysis. Translates technical details
    into business-facing narratives while maintaining technical credibility and metric integrity.
    """

    def __init__(
        self,
        narrative_generator: ProjectNarrativeGenerator | None = None,
        recommendation_layer: ProjectRecommendationLayer | None = None,
    ) -> None:
        self.narrative_generator = narrative_generator or ProjectNarrativeGenerator()
        self.recommendation_layer = recommendation_layer or ProjectRecommendationLayer()

    def frame_project(self, project_input: Any) -> ProjectFramingResponse:
        """
        Orchestrates the project framing analysis. Accepts dicts or ProjectFramingInput objects.
        """
        # 1. Extract metadata fields
        metadata = _get_val(project_input, "metadata", {})
        meta_dict = {
            "title": _get_val(metadata, "title", "Project"),
            "role": _get_val(metadata, "role", "Developer"),
            "description": _get_val(metadata, "description", "") or "",
        }

        # 2. Extract architecture fields
        architecture = _get_val(project_input, "architecture", {})
        arch_dict = {
            "design_patterns": _get_val(architecture, "design_patterns", []) or [],
            "database_setup": _get_val(architecture, "database_setup", "databases"),
            "hosting_or_cloud": _get_val(architecture, "hosting_or_cloud", "Cloud Platform"),
        }

        # 3. Extract lists
        technologies = _get_val(project_input, "technologies", []) or []
        business_goals = _get_val(project_input, "business_goals", []) or []

        # 4. Generate Recruiter Summary
        recruiter_summary = self.narrative_generator.generate_recruiter_summary(
            metadata=meta_dict,
            business_goals=business_goals,
        )

        # 5. Generate Enterprise Framing
        enterprise_framing = self.narrative_generator.generate_enterprise_framing(
            metadata=meta_dict,
            architecture=arch_dict,
            technologies=technologies,
            business_goals=business_goals,
        )

        # 6. Generate Technical Explanations
        technical_explanation = self.narrative_generator.generate_technical_explanation(
            meta_dict,
            arch_dict,
            technologies,
        )

        # 7. Generate Portfolio Recommendations
        portfolio_recommendations = self.recommendation_layer.generate_portfolio_recommendations(
            technologies=technologies,
            architecture=arch_dict,
            business_goals=business_goals,
        )

        # 8. Anti-Hallucination validation checks
        self._validate_no_hallucinations(
            input_desc=meta_dict["description"],
            input_goals=business_goals,
            generated_summary=recruiter_summary.summary_text,
            generated_outcomes=recruiter_summary.key_outcomes,
            generated_scalability=enterprise_framing.scalability_narrative,
            generated_integration=enterprise_framing.integration_narrative,
            generated_impact=enterprise_framing.operational_impact,
            generated_decisions=technical_explanation.architectural_decisions,
            generated_solving=technical_explanation.problem_solving,
        )

        # 9. Narrative justification explanation
        explanation = (
            f"The project '{meta_dict['title']}' was framed converting dev-specific text into impact-focused narratives. "
            f"Recruiter-facing summaries focus on business outcomes, while architectural decisions highlight concrete engineering trade-offs. "
            f"All metrics are verified and preserved for absolute credibility."
        )

        return ProjectFramingResponse(
            recruiter_summary=recruiter_summary,
            enterprise_framing=enterprise_framing,
            technical_explanation=technical_explanation,
            portfolio_recommendations=portfolio_recommendations,
            explanation=explanation,
        )

    def _validate_no_hallucinations(
        self,
        input_desc: str,
        input_goals: list[str],
        generated_summary: str,
        generated_outcomes: list[str],
        generated_scalability: str,
        generated_integration: str,
        generated_impact: str,
        generated_decisions: str,
        generated_solving: str,
    ) -> None:
        """
        Validates that generated narratives do not invent new numeric or percentage metrics.
        """
        from src.modules.resume.tailoring import ResumeTransformationPipeline

        orig_metrics = set()
        orig_metrics.update(ResumeTransformationPipeline.extract_metrics(input_desc))
        for goal in input_goals:
            orig_metrics.update(ResumeTransformationPipeline.extract_metrics(goal))

        gen_metrics = set()
        for text in (
            generated_summary,
            generated_scalability,
            generated_integration,
            generated_impact,
            generated_decisions,
            generated_solving,
        ):
            gen_metrics.update(ResumeTransformationPipeline.extract_metrics(text))
        for outcome in generated_outcomes:
            gen_metrics.update(ResumeTransformationPipeline.extract_metrics(outcome))

        new_metrics = gen_metrics - orig_metrics
        for metric in new_metrics:
            # Ignore standard 4-digit calendar years
            if re.match(r"^\b(19\d{2}|20\d{2})\b$", metric):
                continue
            # Ignore C4 model visual advice if matched in visuals text
            if "c4" in metric or "level 2" in metric.lower():
                continue
            # Raise hallucination exception
            raise ValueError(
                f"Anti-Hallucination Violation: Generated narrative contains unverified metric '{metric}' "
                f"not found in your original project description or business goals."
            )
