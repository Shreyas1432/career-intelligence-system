import re
from typing import Any

from src.modules.domain_alignment import DomainAlignmentEngine
from src.modules.job_extraction.schemas import JobDomain
from src.modules.opportunity_ranking.schemas import (
    FactorScores,
    OpportunityRankingResponse,
    RankingReasoning,
    RankingWeights,
    RecommendationCategory,
)
from src.modules.skill_matching import SkillMatchingEngine
from src.modules.sponsorship import SponsorshipPersistenceService, SponsorshipScoringEngine
from src.modules.user_profile.service import UserProfileService


def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


class OpportunityRankingEngine:
    """
    Weighted multi-factor opportunity ranking engine for career intelligence.
    Combines skill match, domain alignment, visa sponsorship, experience relevance,
    and enterprise alignment factors to produce actionable recommendations.
    """

    def __init__(
        self,
        skill_matching_engine: SkillMatchingEngine | None = None,
        domain_alignment_engine: DomainAlignmentEngine | None = None,
        sponsorship_scoring_engine: SponsorshipScoringEngine | None = None,
    ) -> None:
        self.skill_matching_engine = skill_matching_engine or SkillMatchingEngine()
        self.domain_alignment_engine = domain_alignment_engine or DomainAlignmentEngine()
        self.sponsorship_scoring_engine = sponsorship_scoring_engine

    async def rank_opportunity(
        self,
        user_profile: Any,
        job_intelligence: Any,
        weights: RankingWeights | None = None,
        db_session: Any = None,
    ) -> OpportunityRankingResponse:
        """
        Evaluate, rank, and categorize a job posting against a user's profile.
        """
        active_weights = weights or RankingWeights()

        job_title = _get_field(job_intelligence, "title", "")
        company = _get_field(job_intelligence, "company", "")
        raw_description = _get_field(job_intelligence, "raw_content", "") or ""

        # 1. Check Job Avoidance Filter
        if user_profile and UserProfileService.should_avoid_job(
            user_profile, job_title, company, raw_description
        ):
            return self._build_avoid_response(active_weights)

        # 2. Factor: Skill Matching Score (0-100)
        skill_match_res = await self.skill_matching_engine.match_profile_to_job(
            user_profile, job_intelligence
        )
        skill_score = skill_match_res.final_score

        # 3. Factor: Domain Alignment Score (0-100)
        user_positioning = _get_field(user_profile, "positioning", {})
        domain_match_res = await self.domain_alignment_engine.align_domain(
            user_positioning, job_intelligence
        )
        domain_score = domain_match_res.final_score

        # 4. Factor: Sponsorship Score (0-100)
        sponsorship_score = await self._evaluate_sponsorship_factor(job_intelligence, db_session)

        # 5. Factor: Experience Relevance Score (0-100)
        exp_score = self._calculate_experience_relevance(user_profile, job_intelligence)

        # 6. Factor: Enterprise Alignment Score (0-100)
        ent_score = self._calculate_enterprise_alignment(user_profile, job_intelligence)

        # 7. Weighted score aggregation
        overall_score = round(
            skill_score * active_weights.skill_matching
            + domain_score * active_weights.domain_alignment
            + sponsorship_score * active_weights.sponsorship_probability
            + exp_score * active_weights.experience_relevance
            + ent_score * active_weights.enterprise_alignment,
            2,
        )

        # 8. Determine Recommendation Category
        if overall_score >= 85.0:
            rec = RecommendationCategory.STRONG_APPLY
        elif overall_score >= 65.0:
            rec = RecommendationCategory.APPLY
        elif overall_score >= 40.0:
            rec = RecommendationCategory.WEAK_APPLY
        else:
            rec = RecommendationCategory.SKIP

        # 9. Generate Explainable Reasoning
        factors = FactorScores(
            skill_matching=skill_score,
            domain_alignment=domain_score,
            sponsorship_probability=sponsorship_score,
            experience_relevance=exp_score,
            enterprise_alignment=ent_score,
        )

        reasoning = self._generate_reasoning(factors, rec, overall_score, company)

        return OpportunityRankingResponse(
            overall_score=overall_score,
            recommendation=rec,
            factors=factors,
            weights=active_weights,
            reasoning=reasoning,
        )

    async def _evaluate_sponsorship_factor(self, job_intelligence: Any, db_session: Any) -> float:
        if self.sponsorship_scoring_engine:
            company = _get_field(job_intelligence, "company", "")
            signals = _get_field(job_intelligence, "sponsorship_signals", None)
            res = await self.sponsorship_scoring_engine.evaluate_sponsorship(
                company, extracted_signals=signals
            )
            return float(res.sponsorship_score)

        if db_session:
            company = _get_field(job_intelligence, "company", "")
            signals = _get_field(job_intelligence, "sponsorship_signals", None)
            spon_engine = SponsorshipScoringEngine(SponsorshipPersistenceService(db_session))
            res = await spon_engine.evaluate_sponsorship(company, extracted_signals=signals)
            return float(res.sponsorship_score)

        # Default fallback to JSON database fields if session or engine isn't injected
        signals_dict = _get_field(job_intelligence, "sponsorship_signals", None)
        if isinstance(signals_dict, dict):
            return float(signals_dict.get("score", 50.0))
        return 50.0

    def _calculate_experience_relevance(self, user_profile: Any, job_intelligence: Any) -> float:
        user_positioning = _get_field(user_profile, "positioning", {})
        user_years = _get_field(user_positioning, "years_of_experience", None)
        req_str = _get_field(job_intelligence, "experience_required", "") or ""

        # Extract digits near year keywords
        job_years: int | None = None
        match = re.search(r"(\d+)\s*(?:-\s*(\d+))?\s*(?:year|yr)", req_str, re.IGNORECASE)
        if match:
            job_years = int(match.group(1))
        else:
            match_any = re.search(r"(\d+)", req_str)
            if match_any:
                job_years = int(match_any.group(1))

        seniority_years = {
            "junior": 1,
            "mid": 3,
            "senior": 5,
            "lead": 8,
            "staff": 8,
            "director": 10,
            "principal": 10,
        }

        # Seniority fallback checks
        job_years_val = 3
        if job_years is not None:
            job_years_val = job_years
        else:
            job_title = _get_field(job_intelligence, "title", "") or ""
            title_lower = job_title.lower()
            for sen_name, sen_y in seniority_years.items():
                if sen_name in title_lower:
                    job_years_val = sen_y
                    break

        user_years_val = 3
        if user_years is not None:
            user_years_val = int(user_years)
        else:
            user_seniority = _get_field(user_positioning, "seniority_level", "") or ""
            user_years_val = seniority_years.get(user_seniority.lower(), 3)

        if user_years_val >= job_years_val:
            return 100.0

        # Deduct 20 points per missing year
        return float(max(0.0, 100.0 - (job_years_val - user_years_val) * 20.0))

    def _calculate_enterprise_alignment(self, user_profile: Any, job_intelligence: Any) -> float:
        base_score = 50.0

        job_title = (_get_field(job_intelligence, "title", "") or "").lower()
        target_roles = _get_field(user_profile, "target_roles", [])

        # Check target roles
        role_match = False
        for role in target_roles:
            if role.strip() and role.lower() in job_title:
                role_match = True
                break

        if role_match:
            base_score += 25.0

        # Check target industries / domains
        target_industries = _get_field(user_profile, "target_industries", [])
        job_domain_raw = _get_field(job_intelligence, "domain", JobDomain.UNKNOWN)
        job_domain = (
            job_domain_raw.value if hasattr(job_domain_raw, "value") else str(job_domain_raw)
        )

        ind_match = False
        for ind in target_industries:
            ind_clean = ind.strip().lower()
            if ind_clean and (ind_clean in job_title or ind_clean in job_domain.lower()):
                ind_match = True
                break

        if ind_match:
            base_score += 25.0

        return min(100.0, base_score)

    def _generate_reasoning(
        self,
        factors: FactorScores,
        rec: RecommendationCategory,
        overall_score: float,
        company: str,
    ) -> RankingReasoning:
        strengths = []
        gaps = []

        mapping = {
            "skill_matching": ("technical skill overlap", factors.skill_matching),
            "domain_alignment": ("domain taxonomy alignment", factors.domain_alignment),
            "sponsorship_probability": (
                "visa sponsorship probability",
                factors.sponsorship_probability,
            ),
            "experience_relevance": ("experience requirements fit", factors.experience_relevance),
            "enterprise_alignment": ("enterprise role alignment", factors.enterprise_alignment),
        }

        for _factor_name, (label, score) in mapping.items():
            if score >= 80.0:
                strengths.append(f"Strong {label} (Score: {score}%).")
            elif score < 50.0:
                gaps.append(f"Low {label} (Score: {score}%).")

        explanation = f"Opportunity score of {overall_score}% leads to a '{rec.value.replace('_', ' ').title()}' recommendation. "
        if rec == RecommendationCategory.STRONG_APPLY:
            explanation += f"This role at {company} matches your profile strengths exceptionally well across the board."
        elif rec == RecommendationCategory.APPLY:
            explanation += (
                f"This is a solid target opening at {company} with moderate gaps to address."
            )
        elif rec == RecommendationCategory.WEAK_APPLY:
            explanation += f"You can consider applying to {company}, but be prepared to address the identified alignment gaps."
        else:
            explanation += f"We recommend skipping {company} due to poor matching or overall requirement discrepancies."

        return RankingReasoning(
            strengths=strengths,
            gaps=gaps,
            explanation=explanation,
        )

    def _build_avoid_response(self, weights: RankingWeights) -> OpportunityRankingResponse:
        factors = FactorScores(
            skill_matching=0.0,
            domain_alignment=0.0,
            sponsorship_probability=0.0,
            experience_relevance=0.0,
            enterprise_alignment=0.0,
        )
        reasoning = RankingReasoning(
            strengths=[],
            gaps=["Matches candidate avoidance filters (company/title/keywords)."],
            explanation="Job matches your profile avoidance criteria. Skipping immediately.",
        )
        return OpportunityRankingResponse(
            overall_score=0.0,
            recommendation=RecommendationCategory.SKIP,
            factors=factors,
            weights=weights,
            reasoning=reasoning,
        )
