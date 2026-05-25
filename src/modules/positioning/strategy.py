import re
from typing import Any

from src.modules.positioning.schemas import (
    DifferentiationAnalysis,
    PositioningStatements,
    RecruiterFacingSummary,
    StrategicPositioningResponse,
    ValuePropRecommendation,
)
from src.modules.resume.tailoring import ResumeTransformationPipeline

# Cliché and hype words replaced with objective professional counterparts
HYPE_WORDS_MAP = {
    r"\bthought leader\b": "Specialist",
    r"\bvisionary\b": "Strategist",
    r"\bguru\b": "Expert",
    r"\bninja\b": "Specialist",
    r"\brockstar\b": "Professional",
    r"\bevangelist\b": "Advocate",
    r"\bdisruptor\b": "Innovator",
    r"\bdisruptive\b": "Strategic",
    r"\bgame-changer\b": "Catalyst",
    r"\btransformative\b": "Strategic",
    r"\bcutting-edge\b": "Advanced",
    r"\bworld-class\b": "High-performing",
    r"\bpassionate\b": "Dedicated",
    r"\bsynergy\b": "Collaboration",
}


# ------------------------------------------------------------------------------
# Strategic Differentiation Analyser
# ------------------------------------------------------------------------------

class StrategicDifferentiationAnalyser:
    """
    Analyzes the candidate's unique capabilities, hybrid skill intersections,
    and extracts quantified metrics from projects/work history to serve as core differentiators.
    """

    def _get_skill_combinations(
        self, skills_lower: set[str], profile_skills: list[str]
    ) -> list[str]:
        combinations = []
        has_ai = any(
            s in skills_lower
            for s in ("python", "machine learning", "ml", "ai", "llms", "deep learning")
        )
        has_supply_chain = any(
            s in skills_lower
            for s in ("supply chain", "logistics", "inventory", "warehousing", "s&op")
        )
        has_procurement = any(
            s in skills_lower
            for s in ("procurement", "sourcing", "purchasing", "negotiation", "srm")
        )
        has_consulting = any(
            s in skills_lower
            for s in ("consulting", "strategy", "roadmap", "stakeholder management")
        )
        has_analytics = any(
            s in skills_lower
            for s in ("sql", "analytics", "tableau", "power bi", "dashboards", "excel")
        )

        if has_ai and (has_supply_chain or has_procurement):
            combinations.append("Enterprise AI applied to Supply Chain & Sourcing Logistics")
        if has_ai and has_consulting:
            combinations.append("Strategic Consulting in Enterprise AI Implementation")
        if (has_supply_chain or has_procurement) and has_analytics:
            combinations.append("Operations Analytics in Procurement & Logistics")
        if has_consulting and has_analytics:
            combinations.append("Data-Driven Management Consulting & Analytics")

        # Fallback combination if none match
        if not combinations:
            tech = next(
                (s for s in profile_skills if s.lower() in ("python", "sql", "sap", "oracle")),
                "Technology",
            )
            domain = next(
                (
                    s
                    for s in profile_skills
                    if s.lower() in ("procurement", "logistics", "finance", "operations")
                ),
                "Operations",
            )
            combinations.append(f"{tech}-Driven Solutions for {domain} Operations")
        return combinations

    def _extract_project_differentiators(self, projects: list[dict[str, Any]]) -> list[str]:
        diffs = []
        for proj in projects:
            outcome = proj.get("outcome", "") or ""
            desc = proj.get("description", "") or ""
            metrics_outcome = ResumeTransformationPipeline.extract_metrics(outcome)
            metrics_desc = ResumeTransformationPipeline.extract_metrics(desc)

            if metrics_outcome or metrics_desc:
                title = proj.get("title", "Project")
                val = outcome if metrics_outcome else desc
                if len(val) > 120:
                    val = val[:117] + "..."
                diffs.append(f"Project '{title}' milestone: {val}")
        return diffs

    def _extract_experience_differentiators(self, experiences: list[dict[str, Any]]) -> list[str]:
        diffs = []
        for exp in experiences:
            desc = exp.get("description", "") or ""
            metrics = ResumeTransformationPipeline.extract_metrics(desc)
            if metrics:
                title = exp.get("title", "Role")
                comp = exp.get("company", "Company")
                sentences = desc.split(".")
                matching_sentence = ""
                for s in sentences:
                    if ResumeTransformationPipeline.extract_metrics(s):
                        matching_sentence = s.strip()
                        break
                if not matching_sentence:
                    matching_sentence = desc[:100] + "..."
                diffs.append(f"Experience as '{title}' at '{comp}': {matching_sentence}")
        return diffs

    def analyse_differentiation(
        self,
        profile_skills: list[str],
        experiences: list[dict[str, Any]],
        projects: list[dict[str, Any]],
        opportunity_skills: list[str],
    ) -> DifferentiationAnalysis:
        """
        Runs skill intersection analysis and metric extraction.
        """
        # 1. Unique skill combinations analysis
        skills_lower = {s.lower() for s in profile_skills}
        combinations = self._get_skill_combinations(skills_lower, profile_skills)

        # 2. Core differentiators extraction (quantified metrics)
        differentiators = self._extract_project_differentiators(projects)
        differentiators.extend(self._extract_experience_differentiators(experiences))

        # Fallbacks if no metrics found
        if not differentiators:
            differentiators.append(
                "Proven execution capabilities in technical architecture and team alignment."
            )
            differentiators.append(
                "Hands-on delivery of end-to-end integration and development cycles."
            )

        # 3. Market alignment score calculation
        alignment_score = 80.0
        if opportunity_skills:
            matched = [s for s in opportunity_skills if s.lower() in skills_lower]
            alignment_score = round((len(matched) / len(opportunity_skills)) * 100.0, 2)

        return DifferentiationAnalysis(
            unique_skill_combinations=combinations,
            core_differentiators=differentiators[:4],  # limit to top 4 differentiators
            market_alignment_score=alignment_score,
        )


# ------------------------------------------------------------------------------
# Strategic Narrative Generator
# ------------------------------------------------------------------------------

class StrategicNarrativeGenerator:
    """
    Formulates headlines, elevator pitches, and recruiter summaries.
    Enforces anti-hype filters.
    """

    def clean_hype(self, text: str) -> str:
        """
        Replaces subjective self-promotional terms and clichés with professional equivalents.
        """
        if not text:
            return ""

        cleaned = text
        for pattern, replacement in HYPE_WORDS_MAP.items():

            def replace_match(match: re.Match[str], repl: str = replacement) -> str:
                val = match.group(0)
                if val.istitle():
                    return repl.title()
                if val.isupper():
                    return repl.upper()
                return repl

            cleaned = re.sub(pattern, replace_match, cleaned, flags=re.IGNORECASE)

        return cleaned

    def generate_positioning_statements(
        self, target_roles: list[str], years_of_experience: int | None, style: str
    ) -> PositioningStatements:
        """
        Generates headlines and pitches tailored to the positioning style.
        """
        roles_str = (
            " | ".join(target_roles) if target_roles else "Operations & Technology Professional"
        )
        years_str = f"{years_of_experience}+ years of " if years_of_experience else ""

        style_lower = style.lower()
        if "enterprise_ai" in style_lower or "ai" in style_lower:
            headline = f"{roles_str} | Enterprise AI & Machine Learning Systems"
            pitch = (
                f"Results-driven engineer with {years_str}experience architecting scalable machine learning pipelines, "
                "enterprise model deployments, and data infrastructure."
            )
            focus = [
                "Enterprise AI Architectures",
                "MLOps & CI/CD Pipelines",
                "Large Language Model Integrations",
            ]
        elif "consulting" in style_lower:
            headline = f"{roles_str} | Strategic Operations & Tech Consulting"
            pitch = (
                f"Consultative strategist with {years_str}experience advising leadership, designing roadmaps, "
                "and executing end-to-end digital transformation initiatives."
            )
            focus = [
                "Strategic Digital Roadmap Design",
                "Stakeholder Alignment & Delivery",
                "Process & Tool Optimization",
            ]
        elif "procurement" in style_lower or "supply_chain" in style_lower:
            headline = f"{roles_str} | End-to-End Supply Chain & Sourcing"
            pitch = (
                f"Supply chain specialist with {years_str}experience managing logistics, strategic sourcing, "
                "and vendor relations to achieve cost savings and process efficiency."
            )
            focus = [
                "Strategic Sourcing & Negotiation",
                "Logistics & Warehouse Operations",
                "S&OP & Inventory Optimization",
            ]
        else:
            # operational_analytics
            headline = f"{roles_str} | Operational Analytics & Data Infrastructure"
            pitch = (
                f"Data-driven analyst with {years_str}experience building operational dashboards, KPI metrics, "
                "and automation workflows to drive business efficiency."
            )
            focus = [
                "Operational Dashboard Design",
                "KPI Tracking & Metrics",
                "Data-Driven Process Automation",
            ]

        headline = self.clean_hype(headline)
        pitch = self.clean_hype(pitch)

        return PositioningStatements(headline=headline, elevator_pitch=pitch, focus_areas=focus)

    def generate_recruiter_summary(
        self,
        full_name: str,
        years_of_experience: int | None,
        skills: list[str],
        differentiators: list[str],
        style: str,
    ) -> RecruiterFacingSummary:
        """
        Formulates recruiter-style narrative summary highlighting candidate accomplishments.
        """
        years_str = (
            f"with {years_of_experience}+ years of professional experience"
            if years_of_experience
            else "with a proven track record"
        )
        skills_str = ", ".join(skills[:5])

        style_lower = style.lower()
        if "enterprise_ai" in style_lower or "ai" in style_lower:
            bio = (
                f"{full_name} is an Enterprise AI specialist {years_str}. "
                f"He specializes in building secure, production-ready machine learning solutions, utilizing tools like {skills_str}."
            )
            pillars = ["MLOps Systems", "Data Engineering", "Model Ingestion Pipelines"]
        elif "consulting" in style_lower:
            bio = (
                f"{full_name} is a management consultant {years_str}. "
                f"He partners with corporate leaders to deliver roadmaps and operational improvements, leveraging {skills_str}."
            )
            pillars = ["Stakeholder Alignment", "Value Delivery", "Strategic Advisory"]
        elif "procurement" in style_lower or "supply_chain" in style_lower:
            bio = (
                f"{full_name} is a supply chain operations expert {years_str}. "
                f"He drives cost-reduction sourcing and logistics workflow optimizations, utilizing {skills_str}."
            )
            pillars = ["Strategic Sourcing", "Logistics Management", "Supplier Negotiations"]
        else:
            bio = (
                f"{full_name} is an operational analytics professional {years_str}. "
                f"He translates raw operational data into actionable KPI metrics, utilizing {skills_str}."
            )
            pillars = ["Metric Definition", "Workflow Automation", "Analytics Dashboards"]

        # Synthesize work history impact from differentiators
        if differentiators:
            synthesis = (
                f"Demonstrated career achievements include: {differentiators[0]} "
                f"and {' '.join(differentiators[1:2])}."
            )
        else:
            synthesis = "Proven track record of executing complex technical milestones and delivering business value."

        bio = self.clean_hype(bio)
        synthesis = self.clean_hype(synthesis)

        return RecruiterFacingSummary(
            bio_summary=bio,
            value_pillars=pillars,
            experience_synthesis=synthesis,
        )


# ------------------------------------------------------------------------------
# Strategic Recommendation Layer
# ------------------------------------------------------------------------------

class StrategicRecommendationLayer:
    """
    Formulates targeted strategic recommendations (value props) based on positioning style,
    and maps them to supporting evidence (metrics) found in candidate projects or experiences.
    """

    def generate_recommendations(
        self, style: str, projects: list[dict[str, Any]], experiences: list[dict[str, Any]]
    ) -> list[ValuePropRecommendation]:
        """
        Builds value prop pillars backed by candidate metrics.
        """
        recommendations = []
        style_lower = style.lower()

        # Find supporting metrics from candidate projects/experiences
        supporting_metric = "Proven delivery track record"
        found = False

        # Scan projects first
        for proj in projects:
            outcome = proj.get("outcome", "") or ""
            desc = proj.get("description", "") or ""
            if ResumeTransformationPipeline.extract_metrics(outcome):
                supporting_metric = f"Project '{proj.get('title')}' outcome: {outcome}"
                found = True
                break
            if ResumeTransformationPipeline.extract_metrics(desc):
                supporting_metric = f"Project '{proj.get('title')}' details: {desc}"
                found = True
                break

        # Scan experiences if not found in projects
        if not found:
            for exp in experiences:
                desc = exp.get("description", "") or ""
                if ResumeTransformationPipeline.extract_metrics(desc):
                    supporting_metric = f"Experience at '{exp.get('company')}': {desc[:100]}..."
                    break

        if "enterprise_ai" in style_lower or "ai" in style_lower:
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="Technical Scalability",
                    recommendation_text="Emphasize how you design architectures that handle large data throughput or scale model ingestion workflows.",
                    supporting_evidence=supporting_metric,
                )
            )
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="MLOps Integration",
                    recommendation_text="Focus on your experience setting up CI/CD pipelines, automated testing, and version control for machine learning deployments.",
                    supporting_evidence="Hands-on experience deploying Python and machine learning frameworks.",
                )
            )
        elif "consulting" in style_lower:
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="Business Value Realization",
                    recommendation_text="Highlight how you translate technical initiatives into corporate roadmap milestones and strategic fiscal outcomes.",
                    supporting_evidence=supporting_metric,
                )
            )
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="Executive Alignment",
                    recommendation_text="Emphasize your experience designing status reports and strategic presentations to maintain cross-functional stakeholder alignment.",
                    supporting_evidence="Proven advisory capabilities.",
                )
            )
        elif "procurement" in style_lower or "supply_chain" in style_lower:
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="Fiscal Efficiency & Sourcing",
                    recommendation_text="Highlight negotiation successes, spend category management, and vendor relationship policies to drive savings.",
                    supporting_evidence=supporting_metric,
                )
            )
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="Logistics Process Improvement",
                    recommendation_text="Focus on inventory stock turn rates, warehouse slotting optimizations, and freight transportation metrics.",
                    supporting_evidence="End-to-end logistics coordination experience.",
                )
            )
        else:
            # operational_analytics
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="KPI Metric Design & Visualization",
                    recommendation_text="Emphasize how you define operational KPIs and build self-service dashboards that enable automated, data-driven decisions.",
                    supporting_evidence=supporting_metric,
                )
            )
            recommendations.append(
                ValuePropRecommendation(
                    value_pillar="Process Automation Gains",
                    recommendation_text="Quantify manual work-hours saved by building data pipelines, automated scripts, or database queries.",
                    supporting_evidence="Database querying (SQL) and scripting experience.",
                )
            )

        return recommendations


# ------------------------------------------------------------------------------
# Strategic Positioning Engine
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


class StrategicPositioningEngine:
    """
    Core engine orchestrating career strategic positioning statements, recruiter summaries,
    differentiation analyses, and action-oriented value proposition suggestions.
    """

    def __init__(
        self,
        differentiation_analyser: StrategicDifferentiationAnalyser | None = None,
        narrative_generator: StrategicNarrativeGenerator | None = None,
        recommendation_layer: StrategicRecommendationLayer | None = None,
    ) -> None:
        self.differentiation_analyser = (
            differentiation_analyser or StrategicDifferentiationAnalyser()
        )
        self.narrative_generator = narrative_generator or StrategicNarrativeGenerator()
        self.recommendation_layer = recommendation_layer or StrategicRecommendationLayer()

    def generate_positioning(
        self,
        user_profile: Any,
        projects: list[Any],
        experience: list[Any] | None = None,
        opportunity_intelligence: Any = None,
    ) -> StrategicPositioningResponse:
        """
        Orchestrates the strategic positioning analysis.
        Accepts dicts, Pydantic models, or database ORMs.
        """
        # 1. Safely extract User Profile fields
        full_name = _get_val(user_profile, "full_name", "Technology Professional")
        user_positioning = _get_val(user_profile, "positioning", {})
        years_of_experience = None
        if user_positioning:
            years_of_experience = _get_val(user_positioning, "years_of_experience", None)

        # Extract skills
        raw_skills = _get_val(user_profile, "skills", [])
        if isinstance(raw_skills, str):
            profile_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        elif isinstance(raw_skills, list):
            profile_skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        else:
            profile_skills = []

        domains = _get_val(user_profile, "domains", []) or []

        # 2. Extract Experiences
        experiences_input = (
            experience if experience is not None else _get_val(user_profile, "experience", []) or []
        )
        parsed_experiences = []
        for exp in experiences_input:
            parsed_experiences.append(
                {
                    "title": _get_val(exp, "title", ""),
                    "company": _get_val(exp, "company", ""),
                    "description": _get_val(exp, "description", "") or "",
                }
            )

        # 3. Extract Projects
        parsed_projects = []
        for proj in projects:
            parsed_projects.append(
                {
                    "title": _get_val(proj, "title", ""),
                    "description": _get_val(proj, "description", "") or "",
                    "technologies": _get_val(proj, "technologies", []) or [],
                    "outcome": _get_val(proj, "outcome", "") or "",
                }
            )

        # 4. Resolve target positioning style
        style = self._resolve_positioning_style(opportunity_intelligence, domains, user_profile)

        # 5. Extract opportunity skills
        opp_skills = []
        if opportunity_intelligence:
            raw_opp_skills = _get_val(opportunity_intelligence, "normalized_skills", []) or []
            opp_skills = [str(s) for s in raw_opp_skills]

        # 6. Run Differentiation Analysis
        diff_analysis = self.differentiation_analyser.analyse_differentiation(
            profile_skills=profile_skills,
            experiences=parsed_experiences,
            projects=parsed_projects,
            opportunity_skills=opp_skills,
        )

        # 7. Run Narrative Generation
        statements = self.narrative_generator.generate_positioning_statements(
            target_roles=_get_val(user_profile, "target_roles", []),
            years_of_experience=years_of_experience,
            style=style,
        )

        recruiter_summary = self.narrative_generator.generate_recruiter_summary(
            full_name=full_name,
            years_of_experience=years_of_experience,
            skills=profile_skills,
            differentiators=diff_analysis.core_differentiators,
            style=style,
        )

        # 8. Run Recommendations Layer
        val_recs = self.recommendation_layer.generate_recommendations(
            style=style,
            projects=parsed_projects,
            experiences=parsed_experiences,
        )

        # 9. Anti-Hallucination validation checks
        self._validate_no_hallucinations(
            original_projects=parsed_projects,
            original_experiences=parsed_experiences,
            generated_headline=statements.headline,
            generated_pitch=statements.elevator_pitch,
            generated_bio=recruiter_summary.bio_summary,
            generated_synthesis=recruiter_summary.experience_synthesis,
            years_of_experience=years_of_experience,
        )

        # 10. Generate Engine Explanation Text
        explanation = (
            f"The Strategic Positioning Engine analyzed your profile under the target style '{style.upper()}'. "
            f"Your market alignment score is calculated at {diff_analysis.market_alignment_score}%. "
            f"The positioning statements highlight your core technical strengths while replacing subjective marketing clichés. "
            f"All experience and project highlights are strictly validated to protect quantified impact integrity."
        )

        return StrategicPositioningResponse(
            positioning_statements=statements,
            differentiation=diff_analysis,
            recruiter_summary=recruiter_summary,
            value_prop_recommendations=val_recs,
            explanation=explanation,
        )

    def _resolve_from_opportunity(self, opportunity_intelligence: Any) -> str | None:
        if opportunity_intelligence:
            title = _get_val(opportunity_intelligence, "title", "").lower()
            desc = (
                _get_val(opportunity_intelligence, "raw_content", "").lower()
                or _get_val(opportunity_intelligence, "description", "").lower()
            )

            if (
                "ai" in title
                or "machine learning" in title
                or "ml" in title
                or "ai" in desc
                or "machine learning" in desc
            ):
                return "enterprise_ai"
            if (
                "consultant" in title
                or "consulting" in title
                or "strategy" in title
                or "advisor" in title
            ):
                return "consulting"
            if (
                "procurement" in title
                or "supply chain" in title
                or "logistics" in title
                or "sourcing" in title
                or "procurement" in desc
                or "supply chain" in desc
            ):
                return "procurement_supply_chain"
            if (
                "analyst" in title
                or "analytics" in title
                or "dashboard" in title
                or "reporting" in title
            ):
                return "operational_analytics"
        return None

    def _resolve_from_profile(self, domains: list[str], user_profile: Any) -> str | None:
        # Check profile domains fallback
        domains_lower = {d.lower() for d in domains}
        if "data & ai" in domains_lower or "artificial intelligence" in domains_lower:
            return "enterprise_ai"
        if "consulting" in domains_lower or "advisory" in domains_lower:
            return "consulting"
        if "supply chain" in domains_lower or "procurement" in domains_lower:
            return "procurement_supply_chain"

        # Check target roles fallback
        target_roles = _get_val(user_profile, "target_roles", [])
        if target_roles:
            target_str = " ".join(target_roles).lower()
            if "ai" in target_str or "machine learning" in target_str:
                return "enterprise_ai"
            if "consultant" in target_str:
                return "consulting"
            if (
                "procurement" in target_str
                or "supply chain" in target_str
                or "logistics" in target_str
            ):
                return "procurement_supply_chain"
        return None

    def _resolve_positioning_style(
        self, opportunity_intelligence: Any, domains: list[str], user_profile: Any
    ) -> str:
        """
        Determines the positioning style using target role info or user profile domains.
        Styles: 'enterprise_ai', 'consulting', 'procurement_supply_chain', 'operational_analytics'.
        """
        style = self._resolve_from_opportunity(opportunity_intelligence)
        if style:
            return style

        style = self._resolve_from_profile(domains, user_profile)
        if style:
            return style

        # Default fallback
        return "operational_analytics"

    def _validate_no_hallucinations(
        self,
        original_projects: list[dict[str, Any]],
        original_experiences: list[dict[str, Any]],
        generated_headline: str,
        generated_pitch: str,
        generated_bio: str,
        generated_synthesis: str,
        years_of_experience: int | None,
    ) -> None:
        """
        Validates that generated positioning text doesn't invent new numeric/percentage metrics.
        """
        from src.modules.resume.tailoring import ResumeTransformationPipeline

        orig_metrics = set()
        for proj in original_projects:
            orig_metrics.update(
                ResumeTransformationPipeline.extract_metrics(proj.get("description", "") or "")
            )
            orig_metrics.update(
                ResumeTransformationPipeline.extract_metrics(proj.get("outcome", "") or "")
            )
        for exp in original_experiences:
            orig_metrics.update(
                ResumeTransformationPipeline.extract_metrics(exp.get("description", "") or "")
            )

        # Allow years of experience to be in outputs
        if years_of_experience is not None:
            orig_metrics.add(f"{years_of_experience}")
            orig_metrics.add(f"{years_of_experience}+")

        gen_metrics = set()
        for text in (generated_headline, generated_pitch, generated_bio, generated_synthesis):
            gen_metrics.update(ResumeTransformationPipeline.extract_metrics(text))

        new_metrics = gen_metrics - orig_metrics
        for metric in new_metrics:
            # Ignore standard 4-digit calendar years
            if re.match(r"^\b(19\d{2}|20\d{2})\b$", metric):
                continue
            # Ignore simple experience years matching
            if years_of_experience is not None and (
                metric == str(years_of_experience) or metric == f"{years_of_experience}+"
            ):
                continue
            raise ValueError(
                f"Anti-Hallucination Violation: Generated narrative contains unverified metric '{metric}' "
                f"not found in your experience history or projects list."
            )
