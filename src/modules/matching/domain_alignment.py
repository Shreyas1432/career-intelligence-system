import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.modules.matching.embeddings import EmbeddingPipeline
from src.modules.scraping.normalization import SkillNormalizer, default_skill_normalizer
from src.modules.scraping.schemas import JobDomain, SkillCategory

# ------------------------------------------------------------------------------
# Domain Alignment Schemas
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Domain Alignment Taxonomy
# ------------------------------------------------------------------------------

# Raw keywords for rule-based text matching (matched as whole words or substrings)
DOMAIN_KEYWORDS: dict[DomainCategory, list[str]] = {
    DomainCategory.ENTERPRISE_SYSTEMS: [
        "sap",
        "erp",
        "crm",
        "salesforce",
        "dynamics",
        "workday",
        "oracle erp",
        "netsuite",
        "oracle fusion",
    ],
    DomainCategory.PROCUREMENT: [
        "procurement",
        "sourcing",
        "purchasing",
        "negotiation",
        "spend analytics",
        "category management",
        "srm",
        "vendor management",
        "strategic sourcing",
        "contract management",
        "supplier relationship",
    ],
    DomainCategory.SUPPLY_CHAIN: [
        "supply chain",
        "logistics",
        "inventory",
        "warehousing",
        "demand planning",
        "forecasting",
        "distribution",
        "s&op",
        "fulfillment",
        "freight",
        "transportation",
        "materials management",
    ],
    DomainCategory.AI_ANALYTICS: [
        "machine learning",
        "ml",
        "ai",
        "data science",
        "nlp",
        "llm",
        "generative ai",
        "mlops",
        "python",
        "sql",
        "spark",
        "databricks",
        "analytics",
        "business intelligence",
        "bi",
        "data engineering",
        "dashboard",
        "tableau",
        "power bi",
        "deep learning",
        "neural networks",
    ],
}

# Skill categories mapped to each DomainCategory for skill-based alignment checks
DOMAIN_SKILL_CATEGORIES: dict[DomainCategory, set[SkillCategory]] = {
    DomainCategory.ENTERPRISE_SYSTEMS: {SkillCategory.ENTERPRISE_SYSTEMS},
    DomainCategory.PROCUREMENT: {SkillCategory.PROCUREMENT},
    DomainCategory.SUPPLY_CHAIN: {SkillCategory.SUPPLY_CHAIN},
    DomainCategory.AI_ANALYTICS: {
        SkillCategory.PROGRAMMING,
        SkillCategory.DATA_AI,
        SkillCategory.ANALYTICS,
        SkillCategory.CLOUD_INFRASTRUCTURE,
    },
}


def clean_text_for_matching(text: str) -> str:
    """
    Lowercase and strip punctuation from a string to stabilize keyword lookup.
    """
    if not text:
        return ""
    lowered = text.lower()
    # Replace non-word chars (excluding & and +) with spaces
    cleaned = re.sub(r"[^\w\s&+#]", " ", lowered)
    return " ".join(cleaned.split())


def extract_matched_keywords(text: str, domain: DomainCategory) -> list[str]:
    """
    Search cleaned text for matched keywords associated with a domain.
    Matches keywords either as boundary-isolated words or exact phrases.
    """
    cleaned = clean_text_for_matching(text)
    if not cleaned:
        return []

    matched = []
    for kw in DOMAIN_KEYWORDS[domain]:
        # Formulate regex to match keyword with word boundaries
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, cleaned):
            matched.append(kw)
    return matched


# ------------------------------------------------------------------------------
# Domain Alignment Engine
# ------------------------------------------------------------------------------

def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


class DomainAlignmentEngine:
    """
    Scoring engine to evaluate domain alignment between user positioning and job requirements.
    Combines rule-based keyword/taxonomy overlap and semantic similarity checks.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline | None = None,
        skill_normalizer: SkillNormalizer | None = None,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.skill_normalizer = skill_normalizer or default_skill_normalizer

    async def align_domain(
        self,
        user_positioning: Any,
        job_intelligence: Any,
        similarity_threshold: float = 0.70,
    ) -> DomainAlignmentResponse:
        """
        Evaluate domain alignment between candidate positioning and job requirements.
        """
        positioning_text, headline = self._serialize_user_positioning(user_positioning)
        job_text, job_title, job_skills, job_domain = self._serialize_job_intelligence(
            job_intelligence
        )

        # Calculate Semantic Score
        semantic_sim = 0.0
        if positioning_text and job_text:
            embeddings = await self.embedding_pipeline.embed_texts([positioning_text, job_text])
            if len(embeddings) == 2:
                semantic_sim = self.embedding_pipeline.service.calculate_similarity(
                    embeddings[0], embeddings[1]
                )

        semantic_score = round(max(0.0, semantic_sim * 100.0), 2)

        # Resolve Job Domains
        primary_domains = self._get_primary_domains(job_domain)
        active_domains = self._resolve_active_domains(job_title, job_skills, primary_domains)

        # Calculate Breakdown Scores
        domain_breakdown, all_matched_keywords = self._calculate_domain_scores(
            headline, active_domains, semantic_score
        )

        # Calculate Aggregated Score
        final_score = self._calculate_aggregated_score(
            domain_breakdown, primary_domains, active_domains
        )

        # Generate Explainability Metadata
        strengths, gaps = self._compute_strengths_and_gaps(
            domain_breakdown, active_domains, semantic_sim, similarity_threshold
        )
        explanation = self._generate_explanation_text(final_score, active_domains, domain_breakdown)

        reasoning = ReasoningMetadata(
            semantic_similarity=round(semantic_sim, 4),
            matched_keywords=all_matched_keywords,
            strengths=strengths,
            gaps=gaps,
            explanation=explanation,
        )

        return DomainAlignmentResponse(
            final_score=final_score,
            domain_breakdown=domain_breakdown,
            reasoning=reasoning,
        )

    def _serialize_user_positioning(self, user_positioning: Any) -> tuple[str, str]:
        headline = _get_field(user_positioning, "headline", "")
        seniority = _get_field(user_positioning, "seniority_level", "")
        years_of_exp = _get_field(user_positioning, "years_of_experience", None)

        positioning_parts = []
        if headline:
            positioning_parts.append(headline)
        if seniority:
            positioning_parts.append(f"Seniority: {seniority}")
        if years_of_exp is not None:
            positioning_parts.append(f"Experience: {years_of_exp} years")
        return " | ".join(positioning_parts), headline

    def _serialize_job_intelligence(
        self, job_intelligence: Any
    ) -> tuple[str, str, list[str], JobDomain]:
        job_title = _get_field(job_intelligence, "title", "")
        job_skills = _get_field(job_intelligence, "skills", [])

        job_domain_raw = _get_field(job_intelligence, "domain", JobDomain.UNKNOWN)
        job_domain = JobDomain.UNKNOWN
        if isinstance(job_domain_raw, str):
            try:
                job_domain = JobDomain(job_domain_raw)
            except ValueError:
                job_domain = JobDomain.UNKNOWN
        elif isinstance(job_domain_raw, JobDomain):
            job_domain = job_domain_raw

        job_parts = []
        if job_title:
            job_parts.append(f"Job Title: {job_title}")
        if job_domain != JobDomain.UNKNOWN:
            job_parts.append(f"Domain: {job_domain.value}")
        if job_skills:
            job_parts.append(f"Required Skills: {', '.join(job_skills)}")
        return " | ".join(job_parts), job_title, job_skills, job_domain

    def _calculate_domain_scores(
        self,
        headline: str,
        active_domains: set[DomainCategory],
        semantic_score: float,
    ) -> tuple[dict[DomainCategory, DomainScoreDetails], list[str]]:
        domain_breakdown: dict[DomainCategory, DomainScoreDetails] = {}
        all_matched_keywords = []

        for category in DomainCategory:
            cand_kws = extract_matched_keywords(headline, category)
            all_matched_keywords.extend(cand_kws)

            is_active = category in active_domains
            rule_score = 0.0

            if is_active:
                if cand_kws:
                    rule_score = min(100.0, 50.0 + len(cand_kws) * 25.0)
                else:
                    rule_score = 0.0
            else:
                if cand_kws:
                    rule_score = min(100.0, len(cand_kws) * 40.0)
                else:
                    rule_score = 100.0

            blended_score = round(0.5 * rule_score + 0.5 * semantic_score, 2)

            domain_breakdown[category] = DomainScoreDetails(
                score=blended_score,
                rule_score=round(rule_score, 2),
                semantic_score=semantic_score,
                matched_keywords=cand_kws,
            )

        return domain_breakdown, list(dict.fromkeys(all_matched_keywords))

    def _calculate_aggregated_score(
        self,
        domain_breakdown: dict[DomainCategory, DomainScoreDetails],
        primary_domains: list[DomainCategory],
        active_domains: set[DomainCategory],
    ) -> float:
        total_score_sum = 0.0
        total_weight_sum = 0.0

        for category, details in domain_breakdown.items():
            weight = 0.2
            if category in primary_domains:
                weight = 2.0
            elif category in active_domains:
                weight = 1.0

            total_score_sum += details.score * weight
            total_weight_sum += weight

        return round(total_score_sum / total_weight_sum, 2) if total_weight_sum > 0.0 else 0.0

    def _compute_strengths_and_gaps(
        self,
        domain_breakdown: dict[DomainCategory, DomainScoreDetails],
        active_domains: set[DomainCategory],
        semantic_sim: float,
        similarity_threshold: float,
    ) -> tuple[list[str], list[str]]:
        strengths = []
        gaps = []

        for category, details in domain_breakdown.items():
            is_active = category in active_domains
            if is_active:
                if details.score >= 70.0:
                    strengths.append(
                        f"Strong alignment in active domain '{category.value}' (Score: {details.score}%)."
                    )
                elif details.score < 50.0:
                    gaps.append(
                        f"Missing critical positioning alignment for required domain '{category.value}'."
                    )
            else:
                if 0.0 < details.rule_score < 100.0:
                    strengths.append(
                        f"Transferable expertise detected in '{category.value}' domain."
                    )

        if semantic_sim >= similarity_threshold:
            strengths.append(
                f"Excellent semantic positioning match (Similarity: {round(semantic_sim, 2)})."
            )
        elif semantic_sim < 0.50:
            gaps.append("Overall semantic profile relevance to this job is low.")

        return strengths, gaps

    def _get_primary_domains(self, job_domain: JobDomain) -> list[DomainCategory]:
        """
        Map job classification domains to taxonomy categories.
        """
        if job_domain == JobDomain.OPERATIONS:
            return [DomainCategory.SUPPLY_CHAIN, DomainCategory.PROCUREMENT]
        elif job_domain in (
            JobDomain.DATA_AI,
            JobDomain.SOFTWARE_ENGINEERING,
            JobDomain.INFRASTRUCTURE,
        ):
            return [DomainCategory.AI_ANALYTICS]
        return []

    def _resolve_active_domains(
        self,
        job_title: str,
        job_skills: list[str],
        primary_domains: list[DomainCategory],
    ) -> set[DomainCategory]:
        """
        Determine which domains are explicitly required by checking keywords, skills, and classification.
        """
        active = set(primary_domains)

        # Check job title keywords
        for category in DomainCategory:
            if extract_matched_keywords(job_title, category):
                active.add(category)

        # Check job skills
        for skill in job_skills:
            normalized = self.skill_normalizer.normalize(skill)
            if normalized is not None:
                for category, skill_cats in DOMAIN_SKILL_CATEGORIES.items():
                    if normalized.category in skill_cats:
                        active.add(category)

        return active

    def _generate_explanation_text(
        self,
        final_score: float,
        active_domains: set[DomainCategory],
        domain_breakdown: dict[DomainCategory, DomainScoreDetails],
    ) -> str:
        """
        Generate explainability narrative.
        """
        if final_score >= 80.0:
            fit_label = "exceptional"
        elif final_score >= 60.0:
            fit_label = "strong"
        elif final_score >= 40.0:
            fit_label = "moderate"
        else:
            fit_label = "low"

        active_str = ", ".join(d.value for d in active_domains) if active_domains else "none"

        explanation = (
            f"The candidate's career positioning has a {fit_label} domain alignment ({final_score}%) with the requirements of this role. "
            f"Active domains required by this job were identified as: [{active_str}]. "
        )

        strengths_list = []
        for cat, details in domain_breakdown.items():
            if cat in active_domains and details.score >= 70.0:
                strengths_list.append(cat.value)

        if strengths_list:
            explanation += (
                f"Core alignment strengths include expertise in [{', '.join(strengths_list)}]. "
            )
        else:
            explanation += "There is a lack of deep alignment in the core active domains required. "

        return explanation
