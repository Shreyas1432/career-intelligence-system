import asyncio
import logging
import re
import time
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    and_,
    desc,
    func,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from src.core.database.models import Base, JobIntelligence, UserProfile
from src.core.database.repositories.profile import UserProfileRepository
from src.modules.matching.domain_alignment import (
    DomainAlignmentEngine,
    DomainAlignmentResponse,
    ReasoningMetadata,
)
from src.modules.matching.embeddings import EmbeddingPipeline
from src.modules.matching.explainability import ExplainabilityLayerResponse, ExplainabilityService
from src.modules.matching.skill_matching import (
    ExplainabilityReport,
    ScoreBreakdown,
    SkillMatchingEngine,
    SkillMatchResponse,
)
from src.modules.matching.sponsorship import (
    SponsorshipPersistenceService,
    SponsorshipReasoningMetadata,
    SponsorshipScoringEngine,
    SponsorshipScoringResponse,
    SponsorshipStatus,
)
from src.modules.positioning.profile import UserProfileResponse, UserProfileService
from src.modules.scraping.schemas import JobDomain

logger = logging.getLogger("opportunity_pipeline")


# ------------------------------------------------------------------------------
# Opportunity Ranking Database Model
# ------------------------------------------------------------------------------

class OpportunityRankingResult(Base):
    """
    SQLAlchemy model storing historical opportunity intelligence ranking and scoring results.
    Supports score history, recommendation history, and recalculation tracking.
    """

    __tablename__ = "opportunity_ranking_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("user_profile.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("job_intelligence.id", ondelete="CASCADE"), nullable=False, index=True
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(50), nullable=False)
    factor_scores: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    weights: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    reasoning_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    run_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<OpportunityRankingResult profile_id={self.profile_id} job_id={self.job_id} "
            f"run={self.run_number} score={self.overall_score}>"
        )


# ------------------------------------------------------------------------------
# Opportunity Ranking Schemas
# ------------------------------------------------------------------------------

class RecommendationCategory(StrEnum):
    """
    Actionable recommendation categories.
    """

    STRONG_APPLY = "strong_apply"
    APPLY = "apply"
    WEAK_APPLY = "weak_apply"
    SKIP = "skip"


class RankingWeights(BaseModel):
    """
    Configurable relative factor weights for opportunity ranking.
    """

    model_config = ConfigDict(extra="forbid")

    skill_matching: float = Field(
        default=0.30, ge=0.0, description="Weight of core skill matching (0-1)"
    )
    domain_alignment: float = Field(
        default=0.20, ge=0.0, description="Weight of domain taxonomy alignment (0-1)"
    )
    sponsorship_probability: float = Field(
        default=0.20, ge=0.0, description="Weight of visa sponsorship signals (0-1)"
    )
    experience_relevance: float = Field(
        default=0.15, ge=0.0, description="Weight of experience years/seniority alignment (0-1)"
    )
    enterprise_alignment: float = Field(
        default=0.15, ge=0.0, description="Weight of target roles and industry preferences (0-1)"
    )

    @model_validator(mode="after")
    def normalize_or_validate_weights(self) -> Self:
        """
        Verify that total weights are non-zero and optionally normalize them.
        """
        total = (
            self.skill_matching
            + self.domain_alignment
            + self.sponsorship_probability
            + self.experience_relevance
            + self.enterprise_alignment
        )
        if total <= 0.0:
            raise ValueError("Sum of weights must be greater than zero")

        # We normalize weights to sum exactly to 1.0
        self.skill_matching = round(self.skill_matching / total, 4)
        self.domain_alignment = round(self.domain_alignment / total, 4)
        self.sponsorship_probability = round(self.sponsorship_probability / total, 4)
        self.experience_relevance = round(self.experience_relevance / total, 4)
        self.enterprise_alignment = round(self.enterprise_alignment / total, 4)
        return self


class FactorScores(BaseModel):
    """
    Component scores out of 100 for each evaluated opportunity factor.
    """

    model_config = ConfigDict(extra="forbid")

    skill_matching: float = Field(ge=0.0, le=100.0)
    domain_alignment: float = Field(ge=0.0, le=100.0)
    sponsorship_probability: float = Field(ge=0.0, le=100.0)
    experience_relevance: float = Field(ge=0.0, le=100.0)
    enterprise_alignment: float = Field(ge=0.0, le=100.0)


class RankingReasoning(BaseModel):
    """
    Explainability indicators for opportunity ranking results.
    """

    model_config = ConfigDict(extra="forbid")

    strengths: list[str] = Field(
        default_factory=list, description="Top positive evaluation indicators"
    )
    gaps: list[str] = Field(default_factory=list, description="Critical deficiencies or risks")
    explanation: str = Field(
        description="Paragraph explanation justifying the final recommendation"
    )


class OpportunityRankingResponse(BaseModel):
    """
    Unified ranking response.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    overall_score: float = Field(ge=0.0, le=100.0, description="Blended overall score out of 100")
    recommendation: RecommendationCategory = Field(description="Actionable category mapping")
    factors: FactorScores = Field(description="Calculated factor score breakdown")
    weights: RankingWeights = Field(description="Normalized weights applied")
    reasoning: RankingReasoning = Field(description="Explainability and feedback metadata")


# ------------------------------------------------------------------------------
# Opportunity Ranking Repository
# ------------------------------------------------------------------------------

class OpportunityRankingRepository:
    """
    Data repository for Opportunity Intelligence Ranking and Scoring results.
    Supports score history, recommendation history, profile comparisons, and recalculations.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_ranking_result(
        self,
        profile_id: int,
        job_id: int,
        ranking_response: Any,
    ) -> OpportunityRankingResult:
        """
        Saves a new ranking result, automatically incrementing run_number for recalculation tracking.
        Supports both Pydantic models and raw dictionary payloads.
        """
        if hasattr(ranking_response, "model_dump"):
            data = ranking_response.model_dump()
        elif hasattr(ranking_response, "dict"):
            data = ranking_response.dict()
        else:
            data = ranking_response

        # Determine the next run_number for this profile-job combination
        last_run = (
            self.session.query(OpportunityRankingResult.run_number)
            .filter(
                OpportunityRankingResult.profile_id == profile_id,
                OpportunityRankingResult.job_id == job_id,
            )
            .order_by(desc(OpportunityRankingResult.run_number))
            .first()
        )
        next_run_number = (last_run[0] + 1) if last_run else 1

        db_result = OpportunityRankingResult(
            profile_id=profile_id,
            job_id=job_id,
            overall_score=float(data.get("overall_score", 0.0)),
            recommendation=str(data.get("recommendation", "skip")),
            factor_scores=data.get("factors", {}),
            weights=data.get("weights", {}),
            reasoning_metadata=data.get("reasoning", {}),
            run_number=next_run_number,
        )

        self.session.add(db_result)
        self.session.flush()
        return db_result

    def get_latest_ranking_result(
        self, profile_id: int, job_id: int
    ) -> OpportunityRankingResult | None:
        """
        Gets the latest ranking result for a profile and job comparison.
        """
        return (
            self.session.query(OpportunityRankingResult)
            .filter(
                OpportunityRankingResult.profile_id == profile_id,
                OpportunityRankingResult.job_id == job_id,
            )
            .order_by(desc(OpportunityRankingResult.run_number))
            .first()
        )

    def get_ranking_history(self, profile_id: int, job_id: int) -> list[OpportunityRankingResult]:
        """
        Gets the full recalculation/score history for a profile and job comparison.
        """
        return (
            self.session.query(OpportunityRankingResult)
            .filter(
                OpportunityRankingResult.profile_id == profile_id,
                OpportunityRankingResult.job_id == job_id,
            )
            .order_by(OpportunityRankingResult.run_number)
            .all()
        )

    def get_profile_comparison_history(self, profile_id: int) -> list[OpportunityRankingResult]:
        """
        Gets comparison history of all jobs scored against a profile (latest run per job).
        """
        max_run_subq = (
            self.session.query(
                OpportunityRankingResult.job_id,
                func.max(OpportunityRankingResult.run_number).label("max_run"),
            )
            .filter(OpportunityRankingResult.profile_id == profile_id)
            .group_by(OpportunityRankingResult.job_id)
            .subquery()
        )

        return (
            self.session.query(OpportunityRankingResult)
            .join(
                max_run_subq,
                and_(
                    OpportunityRankingResult.job_id == max_run_subq.c.job_id,
                    OpportunityRankingResult.run_number == max_run_subq.c.max_run,
                ),
            )
            .filter(OpportunityRankingResult.profile_id == profile_id)
            .order_by(desc(OpportunityRankingResult.created_at))
            .all()
        )


# ------------------------------------------------------------------------------
# Opportunity Ranking Engine
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Opportunity Orchestrator Pipeline
# ------------------------------------------------------------------------------

class PipelineContext:
    """
    Shared execution context carrying outputs and status records between pipeline steps.
    """

    def __init__(
        self,
        db_session: Session,
        job_intelligence_id: int,
        profile_id: int | None = None,
        weights: RankingWeights | None = None,
    ):
        self.db_session = db_session
        self.job_intelligence_id = job_intelligence_id
        self.profile_id = profile_id
        self.weights = weights or RankingWeights()

        self.profile: UserProfileResponse | None = None
        self.job_intelligence: JobIntelligence | None = None
        self.profile_embedding: list[float] | None = None
        self.job_embedding: list[float] | None = None
        self.skill_match: SkillMatchResponse | None = None
        self.domain_alignment: DomainAlignmentResponse | None = None
        self.sponsorship: SponsorshipScoringResponse | None = None
        self.ranking: OpportunityRankingResponse | None = None
        self.explainability: ExplainabilityLayerResponse | None = None
        self.ranking_result_id: int | None = None

        self.step_statuses: dict[str, dict[str, Any]] = {}


class OpportunityOrchestrator:
    """
    Lightweight, modular orchestration pipeline for opportunity intelligence analysis.
    Supports structured logging, async-compatible design, and partial failure fallbacks.
    """

    def __init__(self, session: Session):
        self.session = session

    async def run_pipeline(
        self,
        job_intelligence_id: int,
        profile_id: int | None = None,
        weights: RankingWeights | None = None,
    ) -> PipelineContext:
        """
        Executes the entire opportunity analysis flow.
        """
        context = PipelineContext(self.session, job_intelligence_id, profile_id, weights)

        steps = [
            ("profile_loading", self._load_profile),
            ("embedding_generation", self._generate_embeddings),
            ("skill_matching", self._run_skill_matching),
            ("domain_alignment", self._run_domain_alignment),
            ("sponsorship_scoring", self._run_sponsorship_scoring),
            ("ranking", self._run_ranking),
            ("explainability_generation", self._run_explainability),
            ("persistence", self._run_persistence),
        ]

        for step_name, step_func in steps:
            start_time = time.perf_counter()
            logger.info(f"Starting pipeline step: {step_name}")
            try:
                await step_func(context)
                duration = (time.perf_counter() - start_time) * 1000
                context.step_statuses[step_name] = {
                    "status": "success",
                    "duration_ms": round(duration, 2),
                    "error": None,
                }
                logger.info(f"Pipeline step '{step_name}' completed in {duration:.2f}ms")
            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000
                logger.error(f"Pipeline step '{step_name}' failed with error: {e}", exc_info=True)
                context.step_statuses[step_name] = {
                    "status": "failed",
                    "duration_ms": round(duration, 2),
                    "error": str(e),
                }

                if step_name in ("profile_loading", "ranking"):
                    logger.critical(
                        f"Fatal step '{step_name}' failed. Aborting pipeline execution."
                    )
                    break
                else:
                    logger.warning(
                        f"Non-fatal step '{step_name}' failed. Applying fallback values."
                    )
                    self._apply_fallback(context, step_name)

        return context

    async def _load_profile(self, context: PipelineContext) -> None:
        job = (
            context.db_session.query(JobIntelligence)
            .filter(JobIntelligence.id == context.job_intelligence_id)
            .first()
        )
        if not job:
            raise ValueError(f"JobIntelligence with ID {context.job_intelligence_id} not found")
        context.job_intelligence = job

        repo = UserProfileRepository(context.db_session)
        if context.profile_id:
            db_profile = repo.get_by_id(context.profile_id)
        else:
            db_profile = repo.get_active_profile()

        if not db_profile:
            raise ValueError("UserProfile not found in database")

        context.profile = UserProfileResponse.model_validate(db_profile)

    async def _generate_embeddings(self, context: PipelineContext) -> None:
        assert context.job_intelligence is not None
        assert context.profile is not None
        emb_pipeline = EmbeddingPipeline()

        job_title = context.job_intelligence.title or ""
        job_company = context.job_intelligence.company or ""
        job_desc = context.job_intelligence.raw_content or ""

        prof_task = emb_pipeline.embed_profile(context.profile)
        job_task = emb_pipeline.embed_job(job_title, job_company, job_desc)

        context.profile_embedding, context.job_embedding = await asyncio.gather(prof_task, job_task)

    async def _run_skill_matching(self, context: PipelineContext) -> None:
        assert context.profile is not None
        engine = SkillMatchingEngine()
        context.skill_match = await engine.match_profile_to_job(
            context.profile, context.job_intelligence
        )

    async def _run_domain_alignment(self, context: PipelineContext) -> None:
        assert context.profile is not None
        engine = DomainAlignmentEngine()
        context.domain_alignment = await engine.align_domain(
            context.profile.positioning, context.job_intelligence
        )

    async def _run_sponsorship_scoring(self, context: PipelineContext) -> None:
        assert context.job_intelligence is not None
        spon_persistence = SponsorshipPersistenceService(context.db_session)
        engine = SponsorshipScoringEngine(spon_persistence)

        company = context.job_intelligence.company or ""
        signals = context.job_intelligence.sponsorship_signals

        context.sponsorship = await engine.evaluate_sponsorship(company, extracted_signals=signals)

    async def _run_ranking(self, context: PipelineContext) -> None:
        assert context.skill_match is not None
        assert context.domain_alignment is not None
        assert context.sponsorship is not None
        assert context.profile is not None
        assert context.job_intelligence is not None
        ranking_engine = OpportunityRankingEngine()

        skill_score = context.skill_match.final_score
        domain_score = context.domain_alignment.final_score
        sponsorship_score = context.sponsorship.sponsorship_score

        exp_score = ranking_engine._calculate_experience_relevance(
            context.profile, context.job_intelligence
        )
        ent_score = ranking_engine._calculate_enterprise_alignment(
            context.profile, context.job_intelligence
        )

        overall_score = round(
            skill_score * context.weights.skill_matching
            + domain_score * context.weights.domain_alignment
            + sponsorship_score * context.weights.sponsorship_probability
            + exp_score * context.weights.experience_relevance
            + ent_score * context.weights.enterprise_alignment,
            2,
        )

        if overall_score >= 85.0:
            rec = RecommendationCategory.STRONG_APPLY
        elif overall_score >= 65.0:
            rec = RecommendationCategory.APPLY
        elif overall_score >= 40.0:
            rec = RecommendationCategory.WEAK_APPLY
        else:
            rec = RecommendationCategory.SKIP

        job_title = context.job_intelligence.title or ""
        company = context.job_intelligence.company or ""
        raw_description = context.job_intelligence.raw_content or ""

        if UserProfileService.should_avoid_job(
            context.profile, job_title, company, raw_description
        ):
            overall_score = 0.0
            rec = RecommendationCategory.SKIP
            factors = FactorScores(
                skill_matching=0.0,
                domain_alignment=0.0,
                sponsorship_probability=0.0,
                experience_relevance=0.0,
                enterprise_alignment=0.0,
            )
            reasoning = ranking_engine._build_avoid_response(context.weights).reasoning
        else:
            factors = FactorScores(
                skill_matching=skill_score,
                domain_alignment=domain_score,
                sponsorship_probability=sponsorship_score,
                experience_relevance=exp_score,
                enterprise_alignment=ent_score,
            )
            reasoning = ranking_engine._generate_reasoning(factors, rec, overall_score, company)

        context.ranking = OpportunityRankingResponse(
            overall_score=overall_score,
            recommendation=rec,
            factors=factors,
            weights=context.weights,
            reasoning=reasoning,
        )

    async def _run_explainability(self, context: PipelineContext) -> None:
        assert context.job_intelligence is not None
        service = ExplainabilityService()
        company = context.job_intelligence.company or ""
        title = context.job_intelligence.title or ""

        context.explainability = service.generate_explanation(
            skill_match=context.skill_match,
            domain_align=context.domain_alignment,
            sponsorship=context.sponsorship,
            ranking=context.ranking,
            company=company,
            title=title,
        )

    async def _run_persistence(self, context: PipelineContext) -> None:
        assert context.profile is not None
        assert context.job_intelligence is not None
        repo = OpportunityRankingRepository(context.db_session)

        db_prof_id = context.profile_id
        if not db_prof_id:
            db_prof_id = (
                context.db_session.query(UserProfile.id)
                .filter_by(email=context.profile.email)
                .scalar()
            )

        if not db_prof_id:
            raise ValueError("Unable to determine database UserProfile ID for persistence")

        db_result = repo.save_ranking_result(
            profile_id=db_prof_id,
            job_id=context.job_intelligence.id,
            ranking_response=context.ranking,
        )
        context.ranking_result_id = db_result.id

    def _apply_fallback(self, context: PipelineContext, step_name: str) -> None:
        if step_name == "embedding_generation":
            context.profile_embedding = []
            context.job_embedding = []
        elif step_name == "skill_matching":
            context.skill_match = SkillMatchResponse(
                final_score=50.0,
                matched_skills=[],
                missing_skills=[],
                score_breakdown=ScoreBreakdown(
                    exact_match_score=0.0,
                    semantic_match_score=0.0,
                    domain_alignment_bonus=0.0,
                    procurement_supply_chain_bonus=0.0,
                    raw_score=0.0,
                    total_potential_score=0.0,
                    normalized_score=50.0,
                    final_score=50.0,
                ),
                explanation=ExplainabilityReport(
                    summary="Skill matching failed. Defaulting to fallback score.",
                    strengths=[],
                    gaps=[],
                    recommendations=[],
                ),
            )
        elif step_name == "domain_alignment":
            context.domain_alignment = DomainAlignmentResponse(
                final_score=50.0,
                domain_breakdown={},
                reasoning=ReasoningMetadata(
                    semantic_similarity=0.5,
                    matched_keywords=[],
                    strengths=[],
                    gaps=[],
                    explanation="Domain alignment failed. Defaulting to fallback score.",
                ),
            )
        elif step_name == "sponsorship_scoring":
            context.sponsorship = SponsorshipScoringResponse(
                sponsorship_score=50.0,
                sponsorship_confidence=0.5,
                reasoning=SponsorshipReasoningMetadata(
                    historical_approved_petitions=0,
                    historical_denied_petitions=0,
                    extracted_job_status=SponsorshipStatus.UNKNOWN,
                    extracted_job_confidence=0.0,
                    strengths=[],
                    gaps=[],
                    explanation="Sponsorship scoring failed. Defaulting to fallback score.",
                ),
            )
        elif step_name == "explainability_generation":
            context.explainability = ExplainabilityLayerResponse(
                recruiter_summary=(
                    "Orchestration completed with partial failures. Some assessments defaulted."
                ),
                score_composition_explanation=(
                    "Composition math not fully calculated due to step errors."
                ),
                strengths=[],
                weaknesses=["Subsystem evaluation failed"],
                actionable_insights=["Check logs for pipeline step errors."],
                improvement_recommendations=[],
            )
        elif step_name == "persistence":
            pass
