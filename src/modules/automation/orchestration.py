import asyncio
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from src.core.ai.client import ai_client
from src.core.database.models import JobIntelligence
from src.core.prompts import prompt_manager
from src.modules.matching import (
    OpportunityOrchestrator,
    OpportunityRankingResponse,
)
from src.modules.positioning import (
    LinkedInOptimizationEngine,
    LinkedInOptimizationResponse,
    OutreachContextEngine,
    OutreachContextResponse,
    ProjectFramingEngine,
    ProjectFramingResponse,
    StrategicPositioningEngine,
    StrategicPositioningResponse,
    UserProfileResponse,
)
from src.modules.resume import (
    ResumeIntelligenceEngine,
    ResumeTailoringEngine,
    TailoredResumeResponse,
    TailoringStrategyResponse,
)

logger = logging.getLogger("src.modules.automation.orchestration")


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------

class StepExecutionStatus(BaseModel):
    """
    Status tracking data for a single step in the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "failed", "skipped"] = Field(
        description="The outcome state of the step execution"
    )
    duration_ms: float | None = Field(default=None, description="Step duration in milliseconds")
    error_message: str | None = Field(
        default=None, description="Exception error message if execution failed"
    )


class OptimizationPipelineResponse(BaseModel):
    """
    Unified result object representing the outputs of the entire positioning
    and resume optimization orchestration pipeline.
    """

    model_config = ConfigDict(extra="ignore")

    opportunity_ranking: OpportunityRankingResponse = Field(
        description="Fatal core step result: candidate-job match evaluation"
    )
    positioning: StrategicPositioningResponse | None = Field(
        default=None, description="Strategic positioning elevator pitches and bio narratives"
    )
    resume_strategy: TailoringStrategyResponse | None = Field(
        default=None, description="Resume strategy and prioritizations recommendations"
    )
    resume_tailoring: TailoredResumeResponse | None = Field(
        default=None, description="Fully tailored resume and ATS verification results"
    )
    linkedin_optimization: LinkedInOptimizationResponse | None = Field(
        default=None, description="Optimized LinkedIn sections and discoverability indexes"
    )
    outreach_draft: OutreachContextResponse | None = Field(
        default=None, description="Tailored recruiter or networking outreach communications"
    )
    portfolio_framing: ProjectFramingResponse | None = Field(
        default=None, description="Framed technical projects showing business-impact narratives"
    )
    step_statuses: dict[str, StepExecutionStatus] = Field(
        default_factory=dict, description="Metadata mapping step names to execution outcomes"
    )
    explanation: str = Field(
        description="A unified, human-readable summary of the entire pipeline run"
    )


# ------------------------------------------------------------------------------
# Context
# ------------------------------------------------------------------------------

class OptimizationPipelineContext:
    """
    Carries execution outputs, DB references, and statuses through the optimization pipeline.
    """

    def __init__(
        self,
        db_session: Session,
        job_intelligence_id: int,
        profile_id: int | None = None,
        outreach_recipient: dict[str, Any] | None = None,
        outreach_relationship: dict[str, Any] | None = None,
        outreach_preferences: dict[str, Any] | None = None,
        project_to_frame: dict[str, Any] | None = None,
    ):
        self.db_session = db_session
        self.job_intelligence_id = job_intelligence_id
        self.profile_id = profile_id

        # Inputs for optional steps
        self.outreach_recipient = outreach_recipient
        self.outreach_relationship = outreach_relationship
        self.outreach_preferences = outreach_preferences
        self.project_to_frame = project_to_frame

        # Shared loaded ORM objects
        self.profile: UserProfileResponse | None = None
        self.job_intelligence: JobIntelligence | None = None

        # Step outputs
        self.opportunity_ranking: Any = None
        self.positioning: Any = None
        self.resume_strategy: Any = None
        self.resume_tailoring: Any = None
        self.linkedin_optimization: Any = None
        self.outreach_draft: Any = None
        self.portfolio_framing: Any = None

        # Execution records: step_name -> StepExecutionStatus
        self.step_statuses: dict[str, StepExecutionStatus] = {}


# ------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------

class OptimizationPipelineOrchestrator:
    """
    Coordinates modular, async-compatible positioning and resume optimizations.
    Implements structured logging, partial failure handling, and consolidated explainability.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

        # Engine instances
        self.positioning_engine = StrategicPositioningEngine()
        self.resume_intel_engine = ResumeIntelligenceEngine()
        self.resume_tailor_engine = ResumeTailoringEngine()
        self.linkedin_engine = LinkedInOptimizationEngine()
        self.outreach_engine = OutreachContextEngine()
        self.project_engine = ProjectFramingEngine()

    async def run_pipeline(
        self,
        job_intelligence_id: int,
        profile_id: int | None = None,
        outreach_recipient: dict[str, Any] | None = None,
        outreach_relationship: dict[str, Any] | None = None,
        outreach_preferences: dict[str, Any] | None = None,
        project_to_frame: dict[str, Any] | None = None,
    ) -> OptimizationPipelineResponse:
        """
        Executes the entire positioning and optimization pipeline.
        Raises an exception if the fatal opportunity analysis step fails.
        """
        context = OptimizationPipelineContext(
            db_session=self.session,
            job_intelligence_id=job_intelligence_id,
            profile_id=profile_id,
            outreach_recipient=outreach_recipient,
            outreach_relationship=outreach_relationship,
            outreach_preferences=outreach_preferences,
            project_to_frame=project_to_frame,
        )

        steps = [
            ("opportunity_analysis", self._run_opportunity_analysis),
            ("positioning_generation", self._run_positioning_generation),
            ("resume_strategy", self._run_resume_strategy),
            ("resume_tailoring", self._run_resume_tailoring),
            ("linkedin_optimization", self._run_linkedin_optimization),
            ("outreach_preparation", self._run_outreach_preparation),
            ("portfolio_recommendations", self._run_portfolio_recommendations),
        ]

        logger.info("Initializing positioning & optimization orchestration pipeline run...")

        for step_name, step_func in steps:
            start_time = time.perf_counter()
            logger.info(f"Starting pipeline step: '{step_name}'")
            try:
                await step_func(context)
                duration = (time.perf_counter() - start_time) * 1000
                if step_name not in context.step_statuses:
                    context.step_statuses[step_name] = StepExecutionStatus(
                        status="success", duration_ms=round(duration, 2)
                    )
                else:
                    if context.step_statuses[step_name].duration_ms is None:
                        context.step_statuses[step_name].duration_ms = round(duration, 2)
                logger.info(f"Pipeline step '{step_name}' completed in {duration:.2f}ms")
            except Exception as exc:
                duration = (time.perf_counter() - start_time) * 1000
                logger.error(f"Pipeline step '{step_name}' failed with error: {exc}", exc_info=True)
                context.step_statuses[step_name] = StepExecutionStatus(
                    status="failed", duration_ms=round(duration, 2), error_message=str(exc)
                )

                # Abort immediately if the fatal opportunity ranking step fails
                if step_name == "opportunity_analysis":
                    logger.critical(
                        "Fatal opportunity analysis step failed. Aborting pipeline execution."
                    )
                    raise RuntimeError(
                        f"Fatal step '{step_name}' failed: {exc}. Pipeline aborted."
                    ) from exc

        # Generate a unified, human-readable summary explanation
        explanation = self._build_pipeline_explanation(context)

        return OptimizationPipelineResponse(
            opportunity_ranking=context.opportunity_ranking,
            positioning=context.positioning,
            resume_strategy=context.resume_strategy,
            resume_tailoring=context.resume_tailoring,
            linkedin_optimization=context.linkedin_optimization,
            outreach_draft=context.outreach_draft,
            portfolio_framing=context.portfolio_framing,
            step_statuses=context.step_statuses,
            explanation=explanation,
        )

    async def _run_opportunity_analysis(self, context: OptimizationPipelineContext) -> None:
        """
        Executes the fatal opportunity ranking analysis using the underlying OpportunityOrchestrator.
        """
        # Execute the existing opportunity analysis pipeline
        opp_orchestrator = OpportunityOrchestrator(context.db_session)
        opp_context = await opp_orchestrator.run_pipeline(
            job_intelligence_id=context.job_intelligence_id,
            profile_id=context.profile_id,
        )

        # Retrieve loaded entities and outputs from opportunity context
        context.profile = opp_context.profile
        context.job_intelligence = opp_context.job_intelligence
        context.opportunity_ranking = opp_context.ranking

        if not context.opportunity_ranking:
            raise ValueError("Opportunity ranking response was not generated.")

    async def _run_positioning_generation(self, context: OptimizationPipelineContext) -> None:
        """
        Executes StrategicPositioningEngine. Uses profile projects or falls back to empty.
        """
        assert context.profile is not None

        # Extract projects from additional_metadata if available
        metadata_dict = getattr(context.profile, "additional_metadata", {}) or {}
        projects = metadata_dict.get("projects", []) or []

        # Run engine in thread-pool to keep async loop non-blocking
        loop = asyncio.get_running_loop()
        context.positioning = await loop.run_in_executor(
            None,
            self.positioning_engine.generate_positioning,
            context.profile,
            projects,
            getattr(context.profile, "experience", []),
            context.opportunity_ranking,
        )

    async def _run_resume_strategy(self, context: OptimizationPipelineContext) -> None:
        """
        Executes ResumeIntelligenceEngine.
        """
        assert context.profile is not None
        assert context.job_intelligence is not None

        loop = asyncio.get_running_loop()
        context.resume_strategy = await loop.run_in_executor(
            None,
            self.resume_intel_engine.generate_tailoring_strategy,
            context.profile,
            context.job_intelligence,
            context.opportunity_ranking,
        )

    async def _run_resume_tailoring(self, context: OptimizationPipelineContext) -> None:
        """
        Executes ResumeTailoringEngine. Requires strategy output.
        """
        assert context.profile is not None
        assert context.job_intelligence is not None
        assert context.resume_strategy is not None

        loop = asyncio.get_running_loop()
        context.resume_tailoring = await loop.run_in_executor(
            None,
            self.resume_tailor_engine.tailor_resume,
            context.profile,
            context.job_intelligence,
            context.resume_strategy,
            context.opportunity_ranking,
        )

    async def _run_linkedin_optimization(self, context: OptimizationPipelineContext) -> None:
        """
        Executes LinkedInOptimizationEngine.
        """
        assert context.profile is not None
        assert context.job_intelligence is not None

        # Format target roles list
        target_roles = []
        raw_target_roles = getattr(context.profile, "target_roles", []) or []
        if isinstance(raw_target_roles, str):
            target_roles = [r.strip() for r in raw_target_roles.split(",") if r.strip()]
        elif isinstance(raw_target_roles, list):
            target_roles = [str(r).strip() for r in raw_target_roles if str(r).strip()]
        if not target_roles:
            target_roles = [context.job_intelligence.title or "Target Role"]

        # Format trends payload matching expected top_keywords
        trends = {"top_keywords": context.job_intelligence.normalized_skills or []}

        # Dump opportunity ranking to dict to support .get() accesses in sub-engine
        ranking_dict = (
            context.opportunity_ranking.model_dump()
            if hasattr(context.opportunity_ranking, "model_dump")
            else context.opportunity_ranking
        )

        loop = asyncio.get_running_loop()
        context.linkedin_optimization = await loop.run_in_executor(
            None,
            self.linkedin_engine.optimize_profile,
            context.profile,
            target_roles,
            trends,
            ranking_dict,
        )

    async def _run_outreach_preparation(self, context: OptimizationPipelineContext) -> None:
        """
        Executes OutreachContextEngine. Automatically extracts context values if options are missing.
        """
        assert context.profile is not None
        assert context.job_intelligence is not None

        if not context.outreach_recipient:
            logger.info("Outreach recipient details not provided. Skipping outreach preparation.")
            context.step_statuses["outreach_preparation"] = StepExecutionStatus(status="skipped")
            return

        # Fetch communication preferences from profile, fallback to email/formal
        profile_prefs = getattr(context.profile, "communication_preferences", None)
        channel = "email"
        if profile_prefs and getattr(profile_prefs, "channels", None):
            channels = profile_prefs.channels
            if channels and isinstance(channels, list):
                channel = str(channels[0])

        default_pref = {
            "channel": channel,
            "preferred_tone": "formal",
        }
        preferences = context.outreach_preferences or default_pref

        relationship = context.outreach_relationship or {
            "connection_degree": "cold",
            "past_interactions": [],
        }

        # Build opportunity context from active job data
        opportunity = {
            "role_title": context.job_intelligence.title or "Target Role",
            "company": context.job_intelligence.company or "Target Company",
            "key_requirements": context.job_intelligence.normalized_skills or [],
        }

        payload = {
            "recipient": context.outreach_recipient,
            "relationship": relationship,
            "preferences": preferences,
            "opportunity": opportunity,
        }

        loop = asyncio.get_running_loop()
        context.outreach_draft = await loop.run_in_executor(
            None, self.outreach_engine.generate_outreach, payload
        )

    async def _run_portfolio_recommendations(self, context: OptimizationPipelineContext) -> None:
        """
        Executes ProjectFramingEngine. Falls back to profile-registered projects if input is missing.
        """
        assert context.profile is not None

        project_data = context.project_to_frame
        if not project_data:
            # Fallback to the first project in the profile
            metadata_dict = getattr(context.profile, "additional_metadata", {}) or {}
            projects = metadata_dict.get("projects", []) or []
            if projects and isinstance(projects, list):
                project_data = projects[0]

        if not project_data:
            logger.info(
                "No projects provided or found in user profile. Skipping portfolio framing."
            )
            context.step_statuses["portfolio_recommendations"] = StepExecutionStatus(
                status="skipped"
            )
            return

        loop = asyncio.get_running_loop()
        context.portfolio_framing = await loop.run_in_executor(
            None, self.project_engine.frame_project, project_data
        )

    def _build_pipeline_explanation(self, context: OptimizationPipelineContext) -> str:
        """
        Constructs a consolidated human-readable summary of the pipeline results.
        """
        success_steps = [
            name for name, status in context.step_statuses.items() if status.status == "success"
        ]
        failed_steps = [
            name for name, status in context.step_statuses.items() if status.status == "failed"
        ]
        skipped_steps = [
            name for name, status in context.step_statuses.items() if status.status == "skipped"
        ]

        summary = (
            f"Orchestration pipeline execution finished. "
            f"Successfully executed steps: {', '.join(success_steps)}. "
        )
        if failed_steps:
            summary += f"Failed steps: {', '.join(failed_steps)}. "
        if skipped_steps:
            summary += f"Skipped steps: {', '.join(skipped_steps)}. "

        if context.opportunity_ranking:
            score = context.opportunity_ranking.overall_score
            rec = context.opportunity_ranking.recommendation.value
            company_str = (
                f" at {context.job_intelligence.company}"
                if context.job_intelligence and context.job_intelligence.company
                else ""
            )
            summary += (
                f"Candidate matches the opportunity{company_str} with a score of {score:.1f}/100 "
                f"(Decision: {rec.upper()}). "
            )

        if context.resume_tailoring:
            ats_ratio = context.resume_tailoring.ats_metadata.keyword_alignment_ratio
            summary += f"Resume was successfully optimized for ATS with a match ratio of {ats_ratio * 100:.1f}%. "

        return summary


# ------------------------------------------------------------------------------
# Mock Interview & Career Path wrappers (Legacy AI tools compatibility)
# ------------------------------------------------------------------------------

def conduct_mock_interview(role: str, question_type: str, history: list[dict[str, str]]) -> str:
    """
    Handles conversational mock interview generation and feedback evaluation.
    """
    logger.info(f"Generating interview response for {role} (mode: {question_type})...")

    try:
        prompt = prompt_manager.load_prompt(
            "interview/coach_prompt.md",
            {"role": role, "question_type": question_type, "history": str(history)},
        )
    except FileNotFoundError:
        logger.warning("Interview template prompt file not found. Falling back to default string.")
        prompt = (
            f"You are interviewing a candidate for a '{role}' position. "
            f"Context: {question_type}. Generate the next interview statement or feedback response."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a professional, rigorous tech industry interviewer conducting a screen.",
        },
        {"role": "user", "content": prompt},
    ]

    response = ai_client.generate_chat_response(messages)
    return response


def get_career_map(current_role: str, target_role: str, skills: str) -> str:
    """
    Computes a transition roadmap from a current role to a target role, highlighting skill gaps.
    """
    logger.info(f"Generating career transition map from {current_role} to {target_role}...")

    try:
        prompt = prompt_manager.load_prompt(
            "career_path/mapping_prompt.md",
            {"current_role": current_role, "target_role": target_role, "skills": skills},
        )
    except FileNotFoundError:
        logger.warning(
            "Career mapping template prompt file not found. Falling back to default string."
        )
        prompt = (
            f"Map the transition path from '{current_role}' to '{target_role}'. "
            f"Current Skills: {skills}. Identify skills gap, learning path, and estimated timeframes."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a professional career path strategist and tech labor economist.",
        },
        {"role": "user", "content": prompt},
    ]

    response = ai_client.generate_chat_response(messages)
    return response
