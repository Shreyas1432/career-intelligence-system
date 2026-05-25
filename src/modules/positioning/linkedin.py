import re
from typing import Any

from src.modules.positioning.schemas import (
    ImpactLevel,
    LinkedInAboutOptimization,
    LinkedInExperienceOptimization,
    LinkedInHeadlineOptimization,
    LinkedInOptimizationResponse,
    LinkedInOptimizedProfile,
    LinkedInSkillsOptimization,
    ProfileImprovementSuggestion,
    RecruiterKeywordAlignment,
)

# Replaces common resume/social media buzzwords with professional counterparts
BUZZWORDS_MAP = {
    r"\bninja\b": "Specialist",
    r"\bguru\b": "Expert",
    r"\bevangelist\b": "Advocate",
    r"\bdisruptor\b": "Strategist",
    r"\bdisruptive\b": "Strategic",
    r"\bvisionary\b": "Strategist",
    r"\bgame-changer\b": "Catalyst",
    r"\brockstar\b": "Professional",
    r"\btransformative\b": "Strategic",
    r"\bcutting-edge\b": "Advanced",
    r"\bworld-class\b": "High-performing",
    r"\brevolutionizing\b": "Optimizing",
    r"\bsynergy\b": "Collaboration",
    r"\bdisrupting\b": "Optimizing",
    r"\bleading-edge\b": "Advanced",
}


class LinkedInPositioningLayer:
    """
    LinkedIn specific positioning layer. Formulates professional headlines,
    about summaries, and experience description updates while enforcing anti-buzzword constraints.
    """

    def clean_buzzwords(self, text: str) -> str:
        """
        Scans text and replaces generic, non-professional buzzwords with clean,
        industry-standard competencies.
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

    def optimize_headline(
        self,
        original_headline: str,
        target_roles: list[str],
        matched_keywords: list[str],
        _domain: str,
    ) -> tuple[str, str]:
        """
        Constructs an optimized, search-friendly LinkedIn headline (max 220 characters).
        """
        orig_clean = self.clean_buzzwords(original_headline or "")

        # Target roles prefix
        roles_part = " | ".join(target_roles) if target_roles else ""

        # Specific skills/domain suffix
        featured_kws = matched_keywords[:3] if matched_keywords else []
        kws_str = ", ".join(featured_kws)

        parts = []
        if roles_part:
            parts.append(roles_part)
        if kws_str:
            parts.append(f"Specializing in {kws_str}")

        optimized = " | ".join(parts)

        # Fallback to cleaned original if empty
        if not optimized:
            optimized = orig_clean or "Technology & Operations Professional"

        # Apply character cap safely
        if len(optimized) > 220:
            optimized = optimized[:217] + "..."

        optimized = self.clean_buzzwords(optimized)

        justification = (
            f"Emphasized target roles ({', '.join(target_roles)}) and search-trending "
            f"keywords ({', '.join(featured_kws)}) to boost search appearances while "
            f"replacing inflated jargon."
        )

        return optimized, justification

    def optimize_about(
        self,
        _original_about: str,
        _experience_summary: str,
        target_roles: list[str],
        matched_keywords: list[str],
        years_of_experience: int | None,
        domain: str,
    ) -> tuple[str, str]:
        """
        Constructs a structured, compelling About summary incorporating key strengths and trends.
        """
        roles_list = target_roles if target_roles else ["Technology Specialist"]
        years_str = (
            f"with {years_of_experience}+ years of experience "
            if years_of_experience
            else "with extensive experience "
        )

        # 1. Elevator pitch opening
        pitch = (
            f"Result-driven professional {years_str}specializing in {', '.join(roles_list[:2])}."
        )

        # 2. Domain specific focus summary
        dom_lower = domain.lower()
        if "ai" in dom_lower or "analytics" in dom_lower:
            focus = (
                "Focused on architecting enterprise AI systems, building scalable machine learning "
                "data pipelines, and deploying robust models to solve complex business challenges."
            )
        elif "supply" in dom_lower or "procurement" in dom_lower:
            focus = (
                "Focused on optimizing supply chain operations, strategic procurement sourcing, "
                "vendor relationship management (SRM), and executing cost-efficiency strategies."
            )
        else:
            focus = (
                "Focused on leading cross-functional integrations, leveraging technology platforms, "
                "and delivering high-quality business outcomes."
            )

        # 3. Key competencies bulleted block
        expertise_block = ""
        if matched_keywords:
            bullets = [f"• {kw}" for kw in matched_keywords[:6]]
            expertise_block = "Core Competencies:\n" + "\n".join(bullets)

        # Combine sections
        sections = [pitch, focus]
        if expertise_block:
            sections.append(expertise_block)

        optimized = "\n\n".join(sections)
        optimized = self.clean_buzzwords(optimized)

        justification = (
            "Formatted with a standard professional profile structure (opening pitch, domain statement, "
            "and core competence bullet points) to optimize readability and search index matching."
        )

        return optimized, justification

    def optimize_experience_description(
        self,
        _title: str,
        _company: str,
        description: str,
        profile_skills: list[str],
        trending_keywords: list[str],
    ) -> tuple[str, str]:
        """
        Refines work experience descriptions to weave in target keywords naturally, preserving metrics.
        """
        if not description:
            return "", "Empty experience description."

        cleaned_desc = self.clean_buzzwords(description)

        # Find profile skills that are also search trends
        profile_skills_lower = [s.lower() for s in profile_skills]
        relevant_kws = [kw for kw in trending_keywords if kw.lower() in profile_skills_lower]

        # Filter out keywords already mentioned in the description
        missing_in_desc = [kw for kw in relevant_kws if kw.lower() not in cleaned_desc.lower()]

        # Weave up to 2 keywords that the candidate holds
        justifications = []
        if missing_in_desc:
            to_weave = missing_in_desc[:2]
            cleaned_desc += f"\n\nMethodologies and tools utilized: {', '.join(to_weave)}."
            justifications.append(
                f"Weaved in trending technical competencies ({', '.join(to_weave)}) you possess to increase search relevancy."
            )
        else:
            justifications.append(
                "Maintained original description while filtering out marketing buzzwords."
            )

        return cleaned_desc, " ".join(justifications)


class LinkedInRecommendationLayer:
    """
    Computes recruiter search keyword alignment and compiles actionable profile improvement suggestions.
    """

    def calculate_keyword_alignment(
        self,
        profile_skills: list[str],
        trending_keywords: list[str],
        optimized_headline: str,
        optimized_about: str,
    ) -> RecruiterKeywordAlignment:
        """
        Calculates matched/missing trends keywords and outputs search discoverability index.
        """
        if not trending_keywords:
            return RecruiterKeywordAlignment(
                high_priority_keywords=[],
                matched_keywords=[],
                missing_keywords=[],
                discoverability_index=100.0,
            )

        profile_skills_lower = {s.lower() for s in profile_skills}
        headline_lower = optimized_headline.lower()
        about_lower = optimized_about.lower()

        matched_kws = []
        missing_kws = []

        for kw in trending_keywords:
            kw_lower = kw.lower()
            # Deemed matched if it exists in skills list or text of optimized sections
            if (
                kw_lower in profile_skills_lower
                or kw_lower in headline_lower
                or kw_lower in about_lower
            ):
                matched_kws.append(kw)
            else:
                missing_kws.append(kw)

        # Compute index ratio
        ratio = len(matched_kws) / len(trending_keywords)
        discoverability_index = round(ratio * 100.0, 2)

        return RecruiterKeywordAlignment(
            high_priority_keywords=trending_keywords,
            matched_keywords=matched_kws,
            missing_keywords=missing_kws,
            discoverability_index=discoverability_index,
        )

    def generate_improvement_suggestions(
        self,
        profile_skills: list[str],
        trending_keywords: list[str],
        experiences: list[dict[str, Any]],
        opportunity_analysis: dict[str, Any] | None,
    ) -> list[ProfileImprovementSuggestion]:
        """
        Generates actionable suggestions to optimize the candidate's LinkedIn profile layout, content, and search presence.
        """
        suggestions = []

        # 1. Skills section count suggestion
        if len(profile_skills) < 15:
            suggestions.append(
                ProfileImprovementSuggestion(
                    section="Skills",
                    issue=f"Only {len(profile_skills)} skills listed on your profile.",
                    recommendation="Add at least 15-20 skills. LinkedIn search algorithms heavily index the skills section for discoverability.",
                    impact_level=ImpactLevel.HIGH,
                )
            )

        # 2. Short experience descriptions
        for exp in experiences:
            comp = exp.get("company", "Unknown")
            desc = exp.get("description", "") or ""
            if len(desc.strip()) < 50:
                suggestions.append(
                    ProfileImprovementSuggestion(
                        section="Experience",
                        issue=f"Work experience description for '{exp.get('title', 'Role')}' at '{comp}' is too brief.",
                        recommendation="Expand the description to 2-3 bullet points highlighting quantifiable achievements and technologies used.",
                        impact_level=ImpactLevel.HIGH,
                    )
                )

        # 3. Missing critical trending keywords suggestion
        profile_skills_lower = {s.lower() for s in profile_skills}
        missing_trends = [kw for kw in trending_keywords if kw.lower() not in profile_skills_lower]

        if missing_trends:
            kws_to_suggest = missing_trends[:3]
            suggestions.append(
                ProfileImprovementSuggestion(
                    section="Featured Skills",
                    issue=f"High-demand trending keywords ({', '.join(kws_to_suggest)}) are missing from your skills list.",
                    recommendation=f"Add {', '.join(kws_to_suggest)} to your Featured Skills to align with recruiter search criteria.",
                    impact_level=ImpactLevel.MEDIUM,
                )
            )

        # 4. Gaps from opportunity analysis
        if opportunity_analysis:
            reasoning = opportunity_analysis.get("reasoning", {}) or {}
            gaps = reasoning.get("gaps", []) or []
            if gaps:
                top_gap = gaps[0]
                suggestions.append(
                    ProfileImprovementSuggestion(
                        section="Qualifications & Projects",
                        issue=f"Opportunity analysis indicates a gap: {top_gap}",
                        recommendation="Highlight relevant projects or academic work on your LinkedIn profile to bridge this qualification gap.",
                        impact_level=ImpactLevel.MEDIUM,
                    )
                )

        # 5. General Featured Section suggestion
        suggestions.append(
            ProfileImprovementSuggestion(
                section="Featured",
                issue="No links to portfolios, GitHub repositories, or presentations.",
                recommendation="Add a Featured section showing links to your open source projects, tech articles, or portfolio to build social proof.",
                impact_level=ImpactLevel.LOW,
            )
        )

        return suggestions


class LinkedInExplanationLayer:
    """
    Formulates high-level positioning recommendations and compiles a unified,
    human-readable explainability narrative justifying all profile optimization decisions.
    """

    def generate_broad_positioning_recommendations(
        self, target_roles: list[str], discoverability_index: float, domain: str
    ) -> list[str]:
        """
        Creates strategic advice for profile positioning and recruiter engagement.
        """
        recommendations = []
        domain_name = domain or "Technology"

        # 1. Target roles optimization suggestion
        if target_roles:
            recommendations.append(
                f"Position your profile clearly for {', '.join(target_roles)} roles by aligning "
                f"your headline and introduction content to show instant technical relevance."
            )

        # 2. Discoverability thresholds
        if discoverability_index < 60.0:
            recommendations.append(
                "Prioritize updating your Featured Skills section to improve search appearance rates. "
                "Recruiters filter profiles heavily by specific skill tags before reviewing description texts."
            )
        else:
            recommendations.append(
                "Your search discoverability is strong. Maintain engagement by sharing project updates "
                "or professional articles related to your target domain."
            )

        # 3. Domain specific advice
        dom_lower = domain_name.lower()
        if "ai" in dom_lower or "analytics" in dom_lower:
            recommendations.append(
                "Emphasize practical integration experience with Enterprise AI and cloud infrastructure. "
                "Highlight system architecture, MLOps, and model deployment rather than purely academic concepts."
            )
        elif "supply" in dom_lower or "procurement" in dom_lower:
            recommendations.append(
                "Focus on quantifying cost-reduction metrics, vendor negotiations, and end-to-end "
                "logistics optimization. Operations recruiters look for direct fiscal impact statements."
            )
        else:
            recommendations.append(
                "Position as a versatile tech leader by highlighting cross-functional collaborations, "
                "vendor integrations, and modern system architectures."
            )

        # 4. Anti-buzzword reminder
        recommendations.append(
            "Avoid buzzwords like 'ninja', 'guru', or 'disruptive' in your posts or profile. "
            "Recruiters favor objective, evidence-based descriptions of skill and scope."
        )

        return recommendations

    def generate_overall_explanation(
        self,
        headline_justification: str,
        about_justification: str,
        discoverability_index: float,
        domain: str,
    ) -> str:
        """
        Synthesizes a cohesive, recruiter-style summary narrative explaining the optimization choices.
        """
        domain_str = domain or "Technology"

        explanation = (
            f"The LinkedIn profile optimization was successfully formulated focusing on the '{domain_str}' domain. "
            f"The overall search discoverability index stands at {discoverability_index}%. "
            f"Key changes made:\n"
            f"1. Headline: {headline_justification}\n"
            f"2. About Section: {about_justification}\n"
            f"All updates strictly avoid keyword stuffing and clean out inflated buzzwords to present "
            f"an authentic, premium professional persona that appeals to enterprise hiring managers."
        )

        return explanation


def _get_val(obj: Any, field: str, default: Any = None) -> Any:
    """
    Safely retrieves a field value from a dictionary, Pydantic model, or arbitrary object.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


class LinkedInOptimizationEngine:
    """
    Core engine orchestrating LinkedIn profile optimizations. Parses user profiles,
    target roles, job trends, and opportunity comparisons to output optimized sections
    with discoverability statistics and suggestions.
    """

    def __init__(
        self,
        positioning_layer: LinkedInPositioningLayer | None = None,
        recommendation_layer: LinkedInRecommendationLayer | None = None,
        explanation_layer: LinkedInExplanationLayer | None = None,
    ) -> None:
        self.positioning_layer = positioning_layer or LinkedInPositioningLayer()
        self.recommendation_layer = recommendation_layer or LinkedInRecommendationLayer()
        self.explanation_layer = explanation_layer or LinkedInExplanationLayer()

    def optimize_profile(
        self,
        user_profile: Any,
        target_roles: list[str],
        job_intelligence_trends: Any,
        opportunity_analysis: Any = None,
    ) -> LinkedInOptimizationResponse:
        """
        Orchestrates the entire profile optimization analysis.
        Accepts dicts, Pydantic models, or database ORMs.
        """
        # 1. Safely extract User Profile fields
        orig_headline = ""
        years_of_experience = None
        user_positioning = _get_val(user_profile, "positioning", {})
        if user_positioning:
            orig_headline = _get_val(user_positioning, "headline", "")
            years_of_experience = _get_val(user_positioning, "years_of_experience", None)

        orig_about = _get_val(user_profile, "experience_summary", "") or ""
        experiences = _get_val(user_profile, "experience", []) or []

        # Convert experiences list if it contains ORM or Pydantic models
        parsed_experiences = []
        for exp in experiences:
            parsed_experiences.append(
                {
                    "title": _get_val(exp, "title", ""),
                    "company": _get_val(exp, "company", ""),
                    "start_date": _get_val(exp, "start_date", ""),
                    "end_date": _get_val(exp, "end_date", None),
                    "description": _get_val(exp, "description", "") or "",
                }
            )

        # Extract skills
        raw_skills = _get_val(user_profile, "skills", [])
        if isinstance(raw_skills, str):
            profile_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        elif isinstance(raw_skills, list):
            profile_skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        else:
            profile_skills = []

        domains = _get_val(user_profile, "domains", []) or []
        primary_domain = domains[0] if domains else "Technology"

        # 2. Extract Job Trends fields
        trending_keywords = _get_val(job_intelligence_trends, "top_keywords", []) or []

        # 3. Calculate matched/missing keywords before updating sections
        profile_skills_lower = {s.lower() for s in profile_skills}
        matched_kws = [kw for kw in trending_keywords if kw.lower() in profile_skills_lower]

        # 4. Run Positioning Optimizations
        opt_headline_str, headline_just = self.positioning_layer.optimize_headline(
            orig_headline,
            target_roles,
            matched_kws,
            primary_domain,
        )

        opt_about_str, about_just = self.positioning_layer.optimize_about(
            orig_about,
            orig_about,
            target_roles,
            matched_kws,
            years_of_experience,
            primary_domain,
        )

        opt_experiences = []
        for exp in parsed_experiences:
            opt_desc, exp_just = self.positioning_layer.optimize_experience_description(
                exp["title"],
                exp["company"],
                exp["description"],
                profile_skills,
                trending_keywords,
            )
            opt_experiences.append(
                LinkedInExperienceOptimization(
                    title=exp["title"],
                    company=exp["company"],
                    original_description=exp["description"],
                    optimized_description=opt_desc,
                    justification=exp_just,
                )
            )

        # 5. Anti-Hallucination validation checks
        self._validate_no_hallucinations(parsed_experiences, opt_experiences)

        # 6. Skills Recommendations
        missing_trends = [kw for kw in trending_keywords if kw.lower() not in profile_skills_lower]
        skills_opt = LinkedInSkillsOptimization(
            skills_to_add=missing_trends[:5],
            skills_to_remove_or_deprioritize=[],
            justification=(
                f"We recommend adding trending skills ({', '.join(missing_trends[:3])}) you are missing "
                f"to capture recruiters searching for these technical terms in the {primary_domain} market."
            ),
        )

        optimized_profile = LinkedInOptimizedProfile(
            headline=LinkedInHeadlineOptimization(
                original=orig_headline or "",
                optimized=opt_headline_str,
                justification=headline_just,
            ),
            about=LinkedInAboutOptimization(
                original=orig_about, optimized=opt_about_str, justification=about_just
            ),
            experiences=opt_experiences,
            featured_skills=skills_opt,
        )

        # 7. Keyword Alignment & Discoverability index
        keyword_alignment = self.recommendation_layer.calculate_keyword_alignment(
            profile_skills=profile_skills,
            trending_keywords=trending_keywords,
            optimized_headline=opt_headline_str,
            optimized_about=opt_about_str,
        )

        # 8. Improvement Suggestions
        improvement_suggestions = self.recommendation_layer.generate_improvement_suggestions(
            profile_skills=profile_skills,
            trending_keywords=trending_keywords,
            experiences=parsed_experiences,
            opportunity_analysis=opportunity_analysis,
        )

        # 9. High-level recommendations and explainability
        pos_recs = self.explanation_layer.generate_broad_positioning_recommendations(
            target_roles=target_roles,
            discoverability_index=keyword_alignment.discoverability_index,
            domain=primary_domain,
        )

        overall_explanation = self.explanation_layer.generate_overall_explanation(
            headline_justification=headline_just,
            about_justification=about_just,
            discoverability_index=keyword_alignment.discoverability_index,
            domain=primary_domain,
        )

        return LinkedInOptimizationResponse(
            optimized_sections=optimized_profile,
            keyword_alignment=keyword_alignment,
            positioning_recommendations=pos_recs,
            improvement_suggestions=improvement_suggestions,
            explanation=overall_explanation,
        )

    def _validate_no_hallucinations(
        self,
        original_exps: list[dict[str, Any]],
        optimized_exps: list[LinkedInExperienceOptimization],
    ) -> None:
        """
        Validates that optimized experience descriptions do not hallucinate new metrics,
        alter company/title names, or drop existing metrics.
        """
        if len(original_exps) != len(optimized_exps):
            raise ValueError(
                "Anti-Hallucination Guard: Experience count mismatch between original and optimized."
            )

        from src.modules.resume.tailoring import ResumeTransformationPipeline

        for orig, opt in zip(original_exps, optimized_exps, strict=True):
            orig_title = orig.get("title", "")
            opt_title = opt.title
            if orig_title != opt_title:
                raise ValueError(
                    f"Anti-Hallucination Guard: Job title changed from '{orig_title}' to '{opt_title}'."
                )

            orig_company = orig.get("company", "")
            opt_company = opt.company
            if orig_company != opt_company:
                raise ValueError(
                    f"Anti-Hallucination Guard: Company name changed from '{orig_company}' to '{opt_company}'."
                )

            orig_desc = orig.get("description", "") or ""
            opt_desc = opt.optimized_description

            # Check that every original metric is preserved
            orig_metrics = ResumeTransformationPipeline.extract_metrics(orig_desc)
            opt_metrics = ResumeTransformationPipeline.extract_metrics(opt_desc)

            missing_metrics = orig_metrics - opt_metrics
            if missing_metrics:
                raise ValueError(
                    f"Quantified Impact Violation: The metrics {missing_metrics} from role "
                    f"'{opt_title}' at '{opt_company}' were lost during LinkedIn optimization."
                )

            # Check that no new numerical accomplishments are hallucinated
            new_metrics = opt_metrics - orig_metrics
            for metric in new_metrics:
                # Ignore simple 4-digit years (e.g. 2022) to avoid false alerts on dates
                if re.match(r"^\b(19\d{2}|20\d{2})\b$", metric):
                    continue
                raise ValueError(
                    f"Anti-Hallucination Guard: Unverified metric '{metric}' introduced in role "
                    f"'{opt_title}' at '{opt_company}' during LinkedIn optimization."
                )
