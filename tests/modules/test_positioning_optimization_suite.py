from typing import Any

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.automation import (
    OptimizationPipelineOrchestrator,
)
from src.modules.positioning.outreach import OutreachContextEngine
from src.modules.positioning.profile import (
    ExperienceItemSchema,
    PositioningSchema,
    UserProfileCreate,
    UserProfileService,
)
from src.modules.positioning.strategy import StrategicPositioningEngine
from src.modules.scraping.schemas import JobDomain


class MockSentenceTransformerForSuite:
    """
    Mock SentenceTransformer model that returns deterministic unit vectors
    yielding specific cosine similarity values for controlled test cases.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dims = 384

    def encode(
        self,
        texts: str | list[str],
        _convert_to_numpy: bool = True,
        **_kwargs: Any,
    ) -> Any:
        single = isinstance(texts, str)
        texts_list: list[str] = [texts] if isinstance(texts, str) else list(texts)

        results = []
        for text_val in texts_list:
            t = text_val.lower().strip()
            vec = np.zeros(self.dims, dtype=np.float32)
            if "python" in t or "software" in t:
                vec[0] = 1.0
            elif "sql" in t:
                vec[1] = 1.0
            elif (
                "mlops" in t
                or "llms" in t
                or "pytorch" in t
                or "ai" in t
                or "machine learning" in t
            ):
                vec[2] = 1.0
            elif "procurement" in t or "logistics" in t or "sourcing" in t or "supply chain" in t:
                vec[3] = 1.0
            elif "strategy" in t or "consulting" in t or "advisory" in t:
                vec[4] = 1.0
            else:
                # Deterministic random vector based on text hash
                h = hash(text_val)
                np.random.seed(abs(h) % (2**32))
                random_vec = np.random.randn(self.dims).astype(np.float32)
                random_vec[0:10] = 0.0  # Clear controlled indices
                norm = np.linalg.norm(random_vec)
                if norm > 0:
                    random_vec = random_vec / norm
                vec = random_vec
            results.append(vec)

        arr = np.array(results)
        if single:
            return arr[0]
        return arr


@pytest.fixture(autouse=True)
def mock_embedding_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the underlying SentenceTransformer and EmbeddingPipeline
    so all embedding tasks remain fast and deterministic.
    """
    mock_model = MockSentenceTransformerForSuite()
    monkeypatch.setattr(
        "src.modules.matching.embeddings.SentenceTransformer", lambda _name: mock_model
    )

    async def mock_embed_profile(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.1] * 384

    async def mock_embed_job(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.2] * 384

    monkeypatch.setattr(
        "src.modules.matching.EmbeddingPipeline.embed_profile",
        mock_embed_profile,
    )
    monkeypatch.setattr(
        "src.modules.matching.EmbeddingPipeline.embed_job",
        mock_embed_job,
    )


@pytest.fixture(autouse=True)
def enable_foreign_keys(db_session: Session) -> None:
    """
    Ensure SQLite foreign key constraints are enabled.
    """
    db_session.execute(text("PRAGMA foreign_keys=ON;"))


@pytest.fixture
def orchestrator(db_session: Session) -> OptimizationPipelineOrchestrator:
    return OptimizationPipelineOrchestrator(db_session)


@pytest.mark.asyncio
async def test_scenario_enterprise_ai_role(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Enterprise AI Role.
    Candidate profile matches MLOps/LLM requirements, headline optimized for AI,
    anti-hype filters verified, experience prioritization high.
    """
    profile_data = UserProfileCreate(
        full_name="AI Engineer Jane",
        email="jane.ai@example.com",
        skills=["Python", "PyTorch", "AWS", "MLOps", "LLMs"],
        experience_summary="4 years building ML pipelines",
        domains=["data & ai"],
        target_industries=["Technology"],
        target_roles=["Enterprise AI Engineer", "MLOps Architect"],
        positioning=PositioningSchema(
            headline="Machine Learning Developer",
            seniority_level="Mid",
            years_of_experience=4,
        ),
        experience=[
            ExperienceItemSchema(
                title="AI Developer",
                company="AI Labs Inc",
                start_date="2022-01",
                end_date="2026-01",
                description="Built automated pipeline setups. Saved 40% in GPU cloud billing.",
            )
        ],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_ai_hash",
        url="https://example.com/job/ai",
        content_hash="content_ai_hash",
        title="Enterprise AI Platform Architect",
        company="SmartTech Corp",
        location="Remote",
        normalized_skills=["Python", "MLOps", "LLMs", "PyTorch"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="4 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["Python", "MLOps", "LLMs", "PyTorch"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    outreach_rec = {
        "name": "Alex",
        "title": "Engineering Manager",
        "company": "SmartTech Corp",
        "role_type": "engineering_manager",
    }
    project = {
        "metadata": {
            "title": "LLM Inference API",
            "role": "Lead Engineer",
            "description": "High-throughput API endpoints.",
        },
        "architecture": {
            "design_patterns": ["Microservices"],
            "database_setup": "Redis",
            "hosting_or_cloud": "AWS",
        },
        "technologies": ["Python", "PyTorch"],
        "business_goals": ["Reduce latency by 20%"],
    }

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
        outreach_recipient=outreach_rec,
        project_to_frame=project,
    )

    # 1. Pipeline execution status
    assert response.step_statuses["positioning_generation"].status == "success"
    assert response.step_statuses["resume_tailoring"].status == "success"
    assert response.step_statuses["linkedin_optimization"].status == "success"

    # 2. Strategic positioning style and content
    assert response.positioning is not None
    # Verify AI style was resolved
    assert "enterprise_ai" in response.positioning.explanation.lower()

    # 3. LinkedIn Headline contains AI keywords and anti-buzzword filter
    assert response.linkedin_optimization is not None
    headline = response.linkedin_optimization.optimized_sections.headline.optimized
    assert "AI" in headline or "ML" in headline or "Machine Learning" in headline
    # No buzzwords like guru, ninja, disruptor
    assert not any(w in headline.lower() for w in ["guru", "ninja", "disruptor", "rockstar"])

    # 4. Resume Experience Prioritization
    assert response.resume_strategy is not None
    prioritized_exp = response.resume_strategy.prioritized_experiences
    assert len(prioritized_exp) > 0
    # The AI Developer experience should be prioritized high
    assert prioritized_exp[0].company == "AI Labs Inc"
    assert prioritized_exp[0].priority_band.upper() == "HIGH"

    # 5. Project framing details
    assert response.portfolio_framing is not None
    assert "AWS" in response.portfolio_framing.portfolio_recommendations.architecture_visuals_advice


@pytest.mark.asyncio
async def test_scenario_procurement_analytics_role(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Procurement Analytics Role.
    Candidate profile matches logistics, procurement bonus applied,
    outreach adapts to recipient role type.
    """
    profile_data = UserProfileCreate(
        full_name="Supply Expert John",
        email="john.supply@example.com",
        skills=["SQL", "Excel", "Tableau", "Procurement", "Sourcing", "Logistics"],
        experience_summary="6 years analyzing procurement sourcing",
        domains=["supply chain", "procurement"],
        target_industries=["Logistics"],
        target_roles=["Procurement Analyst", "Supply Chain Analyst"],
        positioning=PositioningSchema(
            headline="Sourcing Specialist",
            seniority_level="Senior",
            years_of_experience=6,
        ),
        experience=[
            ExperienceItemSchema(
                title="Procurement Analyst",
                company="Global Sourcing Corp",
                start_date="2020-01",
                end_date="2026-01",
                description="Built automated dashboards. Saved 15% in supply logistics.",
            )
        ],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_procurement_hash",
        url="https://example.com/job/procurement",
        content_hash="content_procurement_hash",
        title="Procurement Sourcing Specialist",
        company="GlobalLogistics Inc",
        location="Remote",
        normalized_skills=["SQL", "Procurement", "Sourcing", "Logistics"],
        domain=JobDomain.OPERATIONS,
        experience_required="5 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["SQL", "Procurement", "Sourcing", "Logistics"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    outreach_rec = {
        "name": "Sarah",
        "title": "Recruiter",
        "company": "GlobalLogistics Inc",
        "role_type": "recruiter",
    }

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
        outreach_recipient=outreach_rec,
    )

    # 1. Pipeline execution status
    assert response.step_statuses["opportunity_analysis"].status == "success"

    # 2. Score check: assert procurement/supply chain domain bonus is applied
    assert response.opportunity_ranking is not None
    # Default operations job maps domain alignment to procurement
    assert response.opportunity_ranking.overall_score >= 65.0

    # 3. Outreach Quality: Recruiter-specific greeting and formal channel tone
    assert response.outreach_draft is not None
    draft_body = response.outreach_draft.draft.body
    assert "Sarah" in draft_body
    # Recommends follow up appropriate for cold recruiting context (10 days)
    assert response.outreach_draft.outreach_recommendations.follow_up_cadence_days == 10


@pytest.mark.asyncio
async def test_scenario_consulting_role(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Consulting Role.
    Candidate profile matches advisory, verifies consulting strategic narrative.
    """
    profile_data = UserProfileCreate(
        full_name="Consultant Clara",
        email="clara@example.com",
        skills=["Strategy", "Financial Modeling", "Management Consulting", "Advisory"],
        experience_summary="8 years consulting enterprise clients",
        domains=["consulting"],
        target_industries=["Management Consulting"],
        target_roles=["Strategy Consultant", "Engagement Manager"],
        positioning=PositioningSchema(
            headline="Advisory Consultant",
            seniority_level="Senior",
            years_of_experience=8,
        ),
        experience=[
            ExperienceItemSchema(
                title="Consultant",
                company="Big Four Advisory",
                start_date="2018-01",
                end_date="2026-01",
                description="Advised Fortune 500 clients. Drove $1M client acquisition value.",
            )
        ],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_consulting_hash",
        url="https://example.com/job/consulting",
        content_hash="content_consulting_hash",
        title="Senior Strategy Consultant",
        company="Apex Advisors",
        location="Remote",
        normalized_skills=["Strategy", "Management Consulting", "Advisory"],
        domain=JobDomain.OTHER,
        experience_required="8 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["Strategy", "Management Consulting", "Advisory"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Verify consulting style resolved
    assert response.positioning is not None
    assert "consulting" in response.positioning.explanation.lower()
    # Confirm advisory value proposition recommendations generated
    assert len(response.positioning.value_prop_recommendations) > 0


@pytest.mark.asyncio
async def test_scenario_underqualified_candidate(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Underqualified Candidate.
    Experience deficit triggers score penalty leading to low match score and SKIP decision.
    """
    profile_data = UserProfileCreate(
        full_name="Junior Dev",
        email="junior@example.com",
        skills=["Python"],
        experience_summary="1 year coding",
        domains=["software_engineering"],
        positioning=PositioningSchema(
            headline="Junior Developer",
            seniority_level="Junior",
            years_of_experience=1,
        ),
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    # Job requires 10 years
    job = JobIntelligence(
        url_hash="job_10yrs_hash",
        url="https://example.com/job/10yrs",
        content_hash="content_10yrs_hash",
        title="Principal Software Architect",
        company="Big Tech Corp",
        location="Remote",
        normalized_skills=["Python", "System Design"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="10 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["Python", "System Design"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Assert penalty triggers a SKIP/WEAK_APPLY decision due to low score
    assert response.opportunity_ranking is not None
    assert response.opportunity_ranking.overall_score < 45.0
    assert response.opportunity_ranking.recommendation.value in ("skip", "weak_apply")


@pytest.mark.asyncio
async def test_scenario_overqualified_candidate(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Overqualified Candidate.
    High seniority candidate is NOT penalized for overqualification.
    """
    profile_data = UserProfileCreate(
        full_name="Principal Guru Jane",
        email="jane.guru@example.com",
        skills=["Python", "System Design"],
        experience_summary="15 years coding",
        domains=["software_engineering"],
        positioning=PositioningSchema(
            headline="Principal Software Engineer",
            seniority_level="Senior",
            years_of_experience=15,
        ),
        experience=[
            ExperienceItemSchema(
                title="Staff Engineer",
                company="Big Corp",
                start_date="2010-01",
                end_date="2025-01",
                description="Built core backend systems using Python.",
            )
        ],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    # Job requires only 2 years
    job = JobIntelligence(
        url_hash="job_2yrs_hash",
        url="https://example.com/job/2yrs",
        content_hash="content_2yrs_hash",
        title="Software Engineer",
        company="Fast Startup",
        location="Remote",
        normalized_skills=["Python"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="2 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["Python"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Overqualified candidate should get a high score and APPLY/STRONG_APPLY decision
    assert response.opportunity_ranking is not None
    assert response.opportunity_ranking.overall_score >= 65.0
    assert response.opportunity_ranking.recommendation.value in ("apply", "strong_apply")


@pytest.mark.asyncio
async def test_scenario_weak_linkedin_profile(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Weak LinkedIn Profile.
    Empty/Seeking headline gets optimized, Featured suggestions returned.
    """
    profile_data = UserProfileCreate(
        full_name="Basic Coder Bob",
        email="bob@example.com",
        skills=["Python"],
        experience_summary="Seeking new opportunities",
        domains=["software_engineering"],
        positioning=PositioningSchema(
            headline="Seeking new opportunities",
            seniority_level="Mid",
            years_of_experience=3,
        ),
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_basic_hash",
        url="https://example.com/job/basic",
        content_hash="content_basic_hash",
        title="Python Engineer",
        company="TechCorp",
        location="Remote",
        normalized_skills=["Python"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="3 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["Python"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # 1. Headline is rewritten professionally, replacing seeking jargon
    assert response.linkedin_optimization is not None
    headline = response.linkedin_optimization.optimized_sections.headline.optimized
    assert "seeking" not in headline.lower()
    assert "opportunities" not in headline.lower()
    assert len(headline) > 0

    # 2. Discoverability suggestions generated
    assert len(response.linkedin_optimization.improvement_suggestions) > 0


@pytest.mark.asyncio
async def test_scenario_strong_positioning_alignment(
    db_session: Session, orchestrator: OptimizationPipelineOrchestrator
) -> None:
    """
    Test scenario: Strong Positioning Alignment.
    Job matches profile perfectly, assertions on ATS match ratio and score thresholds.
    """
    profile_data = UserProfileCreate(
        full_name="Perfect Match Mary",
        email="mary@example.com",
        skills=["Python", "SQL", "Docker"],
        experience_summary="5 years building software",
        domains=["software_engineering"],
        target_industries=["Software"],
        target_roles=["Senior Python Architect", "Software Engineer"],
        positioning=PositioningSchema(
            headline="Senior Software Engineer",
            seniority_level="Senior",
            years_of_experience=5,
        ),
        experience=[
            ExperienceItemSchema(
                title="Software Engineer",
                company="Tech Corp",
                start_date="2020-01",
                end_date="2025-01",
                description="Built Python backend services. Saved $50k in infrastructure costs.",
            )
        ],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_perfect_hash",
        url="https://example.com/job/perfect",
        content_hash="content_perfect_hash",
        title="Senior Python Architect",
        company="TechCorp",
        location="Remote",
        normalized_skills=["Python", "SQL", "Docker"],
        domain=JobDomain.SOFTWARE_ENGINEERING,
        experience_required="5 years",
        sponsorship_signals={"status": "positive", "confidence": 1.0},
    )
    job.skills = ["Python", "SQL", "Docker"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    response = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Assert perfect score, mapping to strong_apply
    assert response.opportunity_ranking is not None
    assert response.opportunity_ranking.overall_score >= 85.0
    assert response.opportunity_ranking.recommendation.value == "strong_apply"

    # ATS Match ratio check
    assert response.resume_tailoring is not None
    assert response.resume_tailoring.ats_metadata.keyword_alignment_ratio >= 0.8


def test_cold_outreach_anti_manipulation_guard() -> None:
    """
    Test that the Outreach Engine's Anti-Manipulation Guard raises a ValueError
    when a cold outreach draft falsely claims pre-existing familiarity.
    """
    engine = OutreachContextEngine()

    # Inputs specifying a cold outreach, but the draft generation helper
    # is mocked/manipulated or we directly check the validator logic
    with pytest.raises(
        ValueError,
        match="Anti-Manipulation Violation: Cold outreach draft contains deceptive phrase",
    ):
        engine._validate_no_manipulation(
            connection_degree="cold",
            draft_body="Hi Alex, it was great connecting with you last week.",
        )

    # Verify that warm outreach allows these phrases
    engine._validate_no_manipulation(
        connection_degree="warm",
        draft_body="Hi Alex, it was great connecting with you last week.",
    )


def test_positioning_anti_hallucination_guard() -> None:
    """
    Test that the Positioning Engine's Anti-Hallucination Guard raises a ValueError
    when generated text invents new numeric or percentage accomplishments.
    """
    engine = StrategicPositioningEngine()

    original_projects = [
        {"title": "API Build", "description": "Built Python endpoints.", "outcome": "Saved $10k"}
    ]
    original_experiences = [
        {"title": "Developer", "company": "Dev LLC", "description": "Increased speed by 15%"}
    ]

    # Valid narratives referencing known metrics or experience years
    engine._validate_no_hallucinations(
        original_projects=original_projects,
        original_experiences=original_experiences,
        generated_headline="Developer with 5 years experience",
        generated_pitch="Built Python endpoints that saved $10k",
        generated_bio="Increased speed by 15% at Dev LLC",
        generated_synthesis="Managed API builds",
        years_of_experience=5,
    )

    # Hallucinated metric "40%" should raise ValueError
    with pytest.raises(
        ValueError,
        match="Anti-Hallucination Violation: Generated narrative contains unverified metric",
    ):
        engine._validate_no_hallucinations(
            original_projects=original_projects,
            original_experiences=original_experiences,
            generated_headline="Developer",
            generated_pitch="Saved $10k and cut latency by 40%",
            generated_bio="Increased speed by 15%",
            generated_synthesis="Managed API builds",
            years_of_experience=5,
        )
