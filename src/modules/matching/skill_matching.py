from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.modules.matching.embeddings import EmbeddingPipeline
from src.modules.scraping.normalization import SkillNormalizer, default_skill_normalizer
from src.modules.scraping.schemas import JobDomain, NormalizedSkill, SkillCategory

# ------------------------------------------------------------------------------
# Skill Matching Schemas
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Skill Scoring Logic
# ------------------------------------------------------------------------------

# Base weights for each SkillCategory bucket
BASE_SKILL_WEIGHTS: dict[SkillCategory, float] = {
    SkillCategory.PROGRAMMING: 1.2,
    SkillCategory.DATA_AI: 1.2,
    SkillCategory.CLOUD_INFRASTRUCTURE: 1.1,
    SkillCategory.ENTERPRISE_SYSTEMS: 1.1,
    SkillCategory.SUPPLY_CHAIN: 1.2,
    SkillCategory.PROCUREMENT: 1.2,
    SkillCategory.SECURITY: 1.1,
    SkillCategory.ANALYTICS: 1.0,
    SkillCategory.PRODUCT: 1.0,
    SkillCategory.BUSINESS: 1.0,
    SkillCategory.OTHER: 0.8,
}

# Domain-specific category multipliers
DOMAIN_MULTIPLIERS: dict[JobDomain, dict[SkillCategory, float]] = {
    JobDomain.SOFTWARE_ENGINEERING: {
        SkillCategory.PROGRAMMING: 1.5,
        SkillCategory.CLOUD_INFRASTRUCTURE: 1.3,
    },
    JobDomain.DATA_AI: {
        SkillCategory.DATA_AI: 1.5,
        SkillCategory.ANALYTICS: 1.3,
    },
    JobDomain.SECURITY: {
        SkillCategory.SECURITY: 1.5,
        SkillCategory.CLOUD_INFRASTRUCTURE: 1.2,
    },
    JobDomain.PRODUCT: {
        SkillCategory.PRODUCT: 1.5,
        SkillCategory.BUSINESS: 1.2,
    },
    JobDomain.OPERATIONS: {
        SkillCategory.SUPPLY_CHAIN: 1.8,
        SkillCategory.PROCUREMENT: 1.8,
        SkillCategory.BUSINESS: 1.2,
        SkillCategory.ANALYTICS: 1.1,
    },
    JobDomain.FINANCE: {
        SkillCategory.BUSINESS: 1.3,
        SkillCategory.ANALYTICS: 1.2,
    },
    JobDomain.SALES: {
        SkillCategory.BUSINESS: 1.4,
    },
    JobDomain.MARKETING: {
        SkillCategory.BUSINESS: 1.3,
        SkillCategory.ANALYTICS: 1.1,
    },
    JobDomain.INFRASTRUCTURE: {
        SkillCategory.CLOUD_INFRASTRUCTURE: 1.5,
        SkillCategory.SECURITY: 1.2,
    },
}


def calculate_skill_weight(category: SkillCategory, domain: JobDomain) -> float:
    """
    Calculate the deterministic weight of a skill based on its category and the job domain.
    """
    base_weight = BASE_SKILL_WEIGHTS.get(category, 1.0)
    multipliers = DOMAIN_MULTIPLIERS.get(domain, {})
    multiplier = multipliers.get(category, 1.0)
    return round(base_weight * multiplier, 2)


def calculate_procurement_bonus(
    job_domain: JobDomain,
    matched_categories: list[SkillCategory],
) -> float:
    """
    Calculate a bonus (up to 10 points) for matching procurement/supply chain skills
    against an operations/procurement role.

    Formula: +2.5 points per matched supply chain or procurement skill, capped at 10.0.
    Only active if job domain is OPERATIONS.
    """
    if job_domain != JobDomain.OPERATIONS:
        return 0.0

    relevant_count = sum(
        1
        for cat in matched_categories
        if cat in (SkillCategory.SUPPLY_CHAIN, SkillCategory.PROCUREMENT)
    )
    return min(10.0, relevant_count * 2.5)


def calculate_domain_alignment_bonus(
    job_domain: JobDomain,
    user_domains: list[str],
    job_title: str | None,
    user_target_industries: list[str],
) -> float:
    """
    Calculate an Enterprise Domain Alignment bonus (up to 10 points).

    - +5 points if the Job's domain matches any of the candidate's target domains (case-insensitive).
    - +5 points if there is a target industry match or domain keyword overlap.
    """
    bonus = 0.0

    # 1. Job domain matching user target domains
    normalized_user_domains = [d.strip().lower().replace("_", " ") for d in user_domains]
    job_domain_str = job_domain.value.lower().replace("_", " ")

    if job_domain_str in normalized_user_domains:
        bonus += 5.0

    # 2. Industry / Title keyword matching
    # Check if the job domain is in target industries, or job title contains target industry words
    if job_title:
        title_lower = job_title.lower()
        for industry in user_target_industries:
            ind_clean = industry.strip().lower()
            if ind_clean and (ind_clean in title_lower or ind_clean in job_domain_str):
                bonus += 5.0
                break

    return min(10.0, bonus)


# ------------------------------------------------------------------------------
# Skill Matching Engine
# ------------------------------------------------------------------------------

def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


class SkillMatchingEngine:
    """
    Production-grade skill matching engine for career intelligence.
    Determines exact and semantic skill alignment, computes domain-specific
    weights, calculates alignment bonuses, and generates explainable feedback.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline | None = None,
        skill_normalizer: SkillNormalizer | None = None,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.skill_normalizer = skill_normalizer or default_skill_normalizer

    async def match_profile_to_job(
        self,
        user_profile: Any,
        job_intelligence: Any,
        similarity_threshold: float = 0.75,
    ) -> SkillMatchResponse:
        """
        Match a user profile against job intelligence.
        Supports both Pydantic schemas and standard dictionary payloads.
        """
        # 1. Extract and normalize skills
        user_skills_raw = _get_field(user_profile, "skills", [])
        job_skills_raw = _get_field(job_intelligence, "skills", [])

        user_normalized = self.skill_normalizer.normalize_many(user_skills_raw)
        job_normalized = self.skill_normalizer.normalize_many(job_skills_raw)

        # Map user normalized skills by canonical lower-case key for exact matching
        user_by_canonical = {ns.canonical.lower(): ns for ns in user_normalized}

        # Determine Job Domain
        job_domain_raw = _get_field(job_intelligence, "domain", JobDomain.UNKNOWN)
        job_domain = JobDomain.UNKNOWN
        if isinstance(job_domain_raw, str):
            try:
                job_domain = JobDomain(job_domain_raw)
            except ValueError:
                job_domain = JobDomain.UNKNOWN
        elif isinstance(job_domain_raw, JobDomain):
            job_domain = job_domain_raw

        # 2. Exact Matching
        matched_details, unmatched_job_skills = self._perform_exact_matching(
            job_normalized, user_by_canonical, job_domain
        )

        # 3. Semantic Matching
        semantic_matches, missing_details = await self._perform_semantic_matching(
            user_normalized, unmatched_job_skills, similarity_threshold
        )
        matched_details.extend(semantic_matches)

        # 4. Score Calculation
        exact_score = sum(md.score for md in matched_details if md.match_type == MatchType.EXACT)
        semantic_score = sum(
            md.score for md in matched_details if md.match_type == MatchType.SEMANTIC
        )

        total_potential_score = sum(md.weight for md in matched_details) + sum(
            ms.weight for ms in missing_details
        )

        if total_potential_score > 0.0:
            normalized_score = round(
                ((exact_score + semantic_score) / total_potential_score) * 100.0, 2
            )
        else:
            normalized_score = 0.0

        # Calculate Bonuses
        user_domains = _get_field(user_profile, "domains", [])
        user_target_industries = _get_field(user_profile, "target_industries", [])
        job_title = _get_field(job_intelligence, "title", "")

        domain_bonus = calculate_domain_alignment_bonus(
            job_domain=job_domain,
            user_domains=user_domains,
            job_title=job_title,
            user_target_industries=user_target_industries,
        )

        matched_categories = []
        for md in matched_details:
            normalized = self.skill_normalizer.normalize(md.matched_skill)
            if normalized is not None:
                matched_categories.append(normalized.category)

        procurement_bonus = calculate_procurement_bonus(
            job_domain=job_domain,
            matched_categories=matched_categories,
        )

        final_score = round(min(100.0, normalized_score + domain_bonus + procurement_bonus), 2)

        score_breakdown = ScoreBreakdown(
            exact_match_score=round(exact_score, 2),
            semantic_match_score=round(semantic_score, 2),
            domain_alignment_bonus=round(domain_bonus, 2),
            procurement_supply_chain_bonus=round(procurement_bonus, 2),
            raw_score=round(exact_score + semantic_score, 2),
            total_potential_score=round(total_potential_score, 2),
            normalized_score=normalized_score,
            final_score=final_score,
        )

        # 5. Generate Explainability Report
        explanation = self._generate_explainability_report(
            final_score=final_score,
            matched_details=matched_details,
            missing_details=missing_details,
            job_title=job_title,
            domain_bonus=domain_bonus,
            procurement_bonus=procurement_bonus,
        )

        return SkillMatchResponse(
            final_score=final_score,
            matched_skills=matched_details,
            missing_skills=missing_details,
            score_breakdown=score_breakdown,
            explanation=explanation,
        )

    def _perform_exact_matching(
        self,
        job_normalized: list[NormalizedSkill],
        user_by_canonical: dict[str, NormalizedSkill],
        job_domain: JobDomain,
    ) -> tuple[list[SkillMatchDetail], list[tuple[NormalizedSkill, float]]]:
        """
        Identify exact matches based on canonical skill representation.
        """
        matched: list[SkillMatchDetail] = []
        unmatched: list[tuple[NormalizedSkill, float]] = []

        for job_ns in job_normalized:
            weight = calculate_skill_weight(job_ns.category, job_domain)
            job_key = job_ns.canonical.lower()

            if job_key in user_by_canonical:
                user_ns = user_by_canonical[job_key]
                matched.append(
                    SkillMatchDetail(
                        matched_skill=job_ns.canonical,
                        user_skill=user_ns.original,
                        job_skill=job_ns.original,
                        match_type=MatchType.EXACT,
                        similarity=1.0,
                        weight=weight,
                        score=round(weight, 2),
                    )
                )
            else:
                unmatched.append((job_ns, weight))

        return matched, unmatched

    async def _perform_semantic_matching(
        self,
        user_normalized: list[NormalizedSkill],
        unmatched_job_skills: list[tuple[NormalizedSkill, float]],
        similarity_threshold: float,
    ) -> tuple[list[SkillMatchDetail], list[MissingSkillDetail]]:
        """
        Identify semantic matches using cosine similarity of skill embeddings.
        """
        matched: list[SkillMatchDetail] = []
        missing: list[MissingSkillDetail] = []

        if not unmatched_job_skills:
            return matched, missing

        if not user_normalized:
            # Candidate has no skills, so everything is missing
            for job_ns, weight in unmatched_job_skills:
                missing.append(
                    MissingSkillDetail(
                        job_skill=job_ns.original,
                        category=job_ns.category,
                        weight=weight,
                    )
                )
            return matched, missing

        # Gather unique canonical skills to generate embeddings
        user_canonicals = list({ns.canonical for ns in user_normalized})
        unmatched_job_canonicals = list({job_ns.canonical for job_ns, _ in unmatched_job_skills})
        all_skills_to_embed = list(set(user_canonicals + unmatched_job_canonicals))

        # Fetch embeddings batch asynchronously
        embeddings = await self.embedding_pipeline.embed_texts(all_skills_to_embed)
        embedding_by_canonical = dict(zip(all_skills_to_embed, embeddings, strict=True))

        for job_ns, weight in unmatched_job_skills:
            job_emb = embedding_by_canonical.get(job_ns.canonical)
            if not job_emb:
                missing.append(
                    MissingSkillDetail(
                        job_skill=job_ns.original,
                        category=job_ns.category,
                        weight=weight,
                    )
                )
                continue

            best_sim = -1.0
            best_user_ns = None

            for user_ns in user_normalized:
                user_emb = embedding_by_canonical.get(user_ns.canonical)
                if not user_emb:
                    continue

                sim = self.embedding_pipeline.service.calculate_similarity(job_emb, user_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_user_ns = user_ns

            if best_sim >= similarity_threshold and best_user_ns is not None:
                matched.append(
                    SkillMatchDetail(
                        matched_skill=job_ns.canonical,
                        user_skill=best_user_ns.original,
                        job_skill=job_ns.original,
                        match_type=MatchType.SEMANTIC,
                        similarity=round(best_sim, 4),
                        weight=weight,
                        score=round(best_sim * weight, 2),
                    )
                )
            else:
                missing.append(
                    MissingSkillDetail(
                        job_skill=job_ns.original,
                        category=job_ns.category,
                        weight=weight,
                    )
                )

        return matched, missing

    def _generate_explainability_report(
        self,
        final_score: float,
        matched_details: list[SkillMatchDetail],
        missing_details: list[MissingSkillDetail],
        job_title: str | None,
        domain_bonus: float,
        procurement_bonus: float,
    ) -> ExplainabilityReport:
        """
        Generate deterministic explainability text, strengths, gaps, and recommendations.
        """
        title = job_title or "this role"

        # Determine verbal fit indicator
        if final_score >= 85.0:
            fit_label = "Excellent Match"
        elif final_score >= 70.0:
            fit_label = "Strong Match"
        elif final_score >= 50.0:
            fit_label = "Good Match with some gaps"
        else:
            fit_label = "Low Alignment"

        exact_count = sum(1 for m in matched_details if m.match_type == MatchType.EXACT)
        semantic_count = sum(1 for m in matched_details if m.match_type == MatchType.SEMANTIC)

        summary = (
            f"The candidate is a '{fit_label}' for the '{title}' role (Score: {final_score}%). "
            f"Matching identified {exact_count} exact skill match(es) and {semantic_count} semantic skill match(es)."
        )

        if domain_bonus > 0.0:
            summary += f" Received +{domain_bonus} Enterprise Domain Alignment bonus."
        if procurement_bonus > 0.0:
            summary += f" Received +{procurement_bonus} Procurement/Supply-Chain Relevance bonus."

        strengths = self._compute_strengths(matched_details)
        gaps = self._compute_gaps(missing_details)
        recommendations = self._compute_recommendations(
            matched_details, missing_details, final_score
        )

        return ExplainabilityReport(
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _compute_strengths(self, matched_details: list[SkillMatchDetail]) -> list[str]:
        strengths = []
        for md in sorted(matched_details, key=lambda x: x.score, reverse=True):
            if md.match_type == MatchType.EXACT:
                strengths.append(f"Strong match in '{md.matched_skill}' (exact match).")
            else:
                pct = int(md.similarity * 100)
                strengths.append(
                    f"Good transferable skill: candidate's '{md.user_skill}' matches required '{md.job_skill}' ({pct}% similarity)."
                )
        return strengths

    def _compute_gaps(self, missing_details: list[MissingSkillDetail]) -> list[str]:
        gaps = []
        for ms in sorted(missing_details, key=lambda x: x.weight, reverse=True):
            gaps.append(f"Missing required '{ms.job_skill}' (Category: {ms.category.value}).")
        return gaps

    def _compute_recommendations(
        self,
        matched_details: list[SkillMatchDetail],
        missing_details: list[MissingSkillDetail],
        final_score: float,
    ) -> list[str]:
        recommendations = []
        for ms in sorted(missing_details, key=lambda x: x.weight, reverse=True)[:3]:
            recommendations.append(
                f"Acquire proficiency in '{ms.job_skill}' to address a key requirement in {ms.category.value}."
            )

        for md in matched_details:
            if md.match_type == MatchType.SEMANTIC and md.similarity >= 0.85:
                recommendations.append(
                    f"Explicitly list '{md.job_skill}' on your resume since you have '{md.user_skill}' ({int(md.similarity * 100)}% match)."
                )

        if not recommendations:
            if final_score >= 85.0:
                recommendations.append(
                    "Your skills match this job perfectly. Tailor your resume summary to highlight your domain alignment."
                )
            else:
                recommendations.append(
                    "Consider adding more domain-specific projects to your profile."
                )

        return list(dict.fromkeys(recommendations))
