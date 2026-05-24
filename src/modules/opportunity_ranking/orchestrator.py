import asyncio
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.core.database.repositories.profile import UserProfileRepository
from src.modules.domain_alignment import DomainAlignmentEngine
from src.modules.domain_alignment.schemas import DomainAlignmentResponse, ReasoningMetadata
from src.modules.embeddings.pipeline import EmbeddingPipeline
from src.modules.explainability import ExplainabilityLayerResponse, ExplainabilityService
from src.modules.opportunity_ranking.engine import OpportunityRankingEngine
from src.modules.opportunity_ranking.repository import OpportunityRankingRepository
from src.modules.opportunity_ranking.schemas import (
    FactorScores,
    OpportunityRankingResponse,
    RankingWeights,
    RecommendationCategory,
)
from src.modules.skill_matching import SkillMatchingEngine
from src.modules.skill_matching.schemas import (
    ExplainabilityReport,
    ScoreBreakdown,
    SkillMatchResponse,
)
from src.modules.sponsorship import SponsorshipPersistenceService, SponsorshipScoringEngine
from src.modules.sponsorship.schemas import (
    SponsorshipReasoningMetadata,
    SponsorshipScoringResponse,
)
from src.modules.sponsorship.types import SponsorshipStatus
from src.modules.user_profile import UserProfileResponse, UserProfileService

logger = logging.getLogger("opportunity_pipeline")


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

        # Pipeline outputs
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

        # Step statuses: step_name -> {"status": "success"/"failed", "duration_ms": float, "error": str}
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

                # Profile loading and ranking are fatal steps; abort if they fail
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
        # Load Job
        job = (
            context.db_session.query(JobIntelligence)
            .filter(JobIntelligence.id == context.job_intelligence_id)
            .first()
        )
        if not job:
            raise ValueError(f"JobIntelligence with ID {context.job_intelligence_id} not found")
        context.job_intelligence = job

        # Load Profile
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

        # Concurrently generate embeddings
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

        # Check avoidance filter override
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
            # Fetch default skipped avoidance reasoning
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
        # Verify both objects have ids (meaning we have DB persistence)
        # Note: context.profile_id or default active profile ID
        from src.core.database.models import UserProfile

        db_prof_id = context.profile_id
        if not db_prof_id:
            db_profile = (
                context.db_session.query(UserProfile.id)
                .filter_by(email=context.profile.email)
                .first()
            )
            db_prof_id = db_profile[0] if db_profile else None

        if not db_prof_id:
            raise ValueError("Unable to determine database UserProfile ID for persistence")

        db_result = repo.save_ranking_result(
            profile_id=db_prof_id,
            job_id=context.job_intelligence.id,
            ranking_response=context.ranking,
        )
        context.ranking_result_id = db_result.id

    def _apply_fallback(self, context: PipelineContext, step_name: str) -> None:
        """
        Applies fallback default data responses when a pipeline step fails.
        """
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
            # Persistence failure does not change context data models, simply log
            pass
