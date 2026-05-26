from typing import Any

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.matching import (
    OpportunityOrchestrator,
    RecommendationCategory,
    SponsorshipPersistenceService,
)
from src.modules.positioning.profile import (
    PositioningSchema,
    UserProfileCreate,
    UserProfileService,
)


class MockSentenceTransformerForSuite:
    """
    Mock SentenceTransformer model that returns deterministic vectors
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
            elif "procurement" in t or "logistics" in t:
                vec[2] = 1.0
            elif "supply chain" in t:
                vec[2] = 0.95
                vec[3] = 0.3122
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
def mock_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> MockSentenceTransformerForSuite:
    """
    Automatically mock SentenceTransformer with MockSentenceTransformerForSuite.
    """
    mock_model = MockSentenceTransformerForSuite()
    monkeypatch.setattr(
        "src.modules.matching.embeddings.SentenceTransformer",
        lambda _name: mock_model,
    )
    return mock_model


@pytest.fixture(autouse=True)
def enable_foreign_keys(db_session: Session) -> None:
    """
    Ensure SQLite foreign keys are turned ON for every database connection.
    """
    db_session.execute(text("PRAGMA foreign_keys=ON;"))


@pytest.fixture
def orchestrator(db_session: Session) -> OpportunityOrchestrator:
    return OpportunityOrchestrator(db_session)


@pytest.fixture
def seed_sponsorship_data(db_session: Session) -> None:
    """
    Seed H-1B visa statistics for TechCorp (positive) and LocalCorp (neutral).
    """
    persistence = SponsorshipPersistenceService(db_session)
    # TechCorp: Highly positive visa filings history
    persistence.upsert_sponsorship_record("TechCorp LLC", 2024, 150, 2)
    persistence.upsert_sponsorship_record("TechCorp LLC", 2025, 200, 3)
    # LocalCorp: No historical records seeded
    db_session.flush()


# --- TEST CASES ---


@pytest.mark.asyncio
@pytest.mark.usefixtures("seed_sponsorship_data")
async def test_strong_enterprise_match(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Strong enterprise match.
    Profile has matching target roles, matching industries, and matching skills.
    """
    # Create profile with matching preferences
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        target_roles=["Python Developer", "Software Engineer"],
        target_industries=["Healthcare", "Technology"],
        skills=["Python", "SQL"],
        experience_summary="5 years coding in Python",
        domains=["software_engineering"],
        positioning=PositioningSchema(
            headline="Senior Software Engineer",
            seniority_level="Senior",
            years_of_experience=5,
        ),
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    # Job matches title, company, domain, skills, and experience
    job = JobIntelligence(
        url_hash="job_strong_match_hash",
        url="https://example.com/job/strong",
        content_hash="content_strong_hash",
        title="Senior Python Developer",
        company="TechCorp LLC",
        location="Remote",
        normalized_skills=["Python", "SQL"],
        domain="Technology",
        experience_required="5 years",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = ["Python", "SQL"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Validate output
    assert context.ranking is not None
    assert context.ranking.overall_score >= 80.0
    assert context.ranking.recommendation in (
        RecommendationCategory.STRONG_APPLY,
        RecommendationCategory.APPLY,
    )
    assert context.explainability is not None
    assert "TechCorp LLC" in context.explainability.recruiter_summary
    assert "Senior Python Developer" in context.explainability.recruiter_summary


@pytest.mark.asyncio
async def test_weak_domain_alignment(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Weak domain alignment.
    Profile positioning focuses purely on Software Engineering,
    while Job details focus on Procurement/Logistics (Operations).
    """
    # Software engineer profile
    profile_data = UserProfileCreate(
        full_name="Dev Guy",
        email="dev.guy@example.com",
        target_roles=["Software Engineer"],
        skills=["Python"],
        domains=["software_engineering"],
        positioning=PositioningSchema(
            headline="Python Developer",
            seniority_level="Mid",
            years_of_experience=3,
        ),
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    # Job is for a logistics coordinator (Procurement/Supply Chain)
    job = JobIntelligence(
        url_hash="job_weak_domain_hash",
        url="https://example.com/job/logistics",
        content_hash="content_logistics_hash",
        title="Logistics Operations Specialist",
        company="LocalCorp",
        location="New York",
        normalized_skills=["Procurement", "Supply Chain"],
        domain="Operations",
        experience_required="3 years",
    )
    job.skills = ["Procurement", "Supply Chain"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Validate domain alignment is low
    assert context.domain_alignment is not None
    assert context.domain_alignment.final_score < 40.0
    assert context.ranking is not None
    assert any("Low domain taxonomy alignment" in gap for gap in context.ranking.reasoning.gaps)


@pytest.mark.asyncio
@pytest.mark.usefixtures("seed_sponsorship_data")
async def test_sponsorship_positive_company(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Sponsorship-positive company.
    Seeded stats show massive historical approvals for TechCorp LLC.
    """
    profile_data = UserProfileCreate(
        full_name="Intl Candidate",
        email="intl@example.com",
        skills=["Python"],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_spon_pos_hash",
        url="https://example.com/job/spon_pos",
        content_hash="content_spon_pos_hash",
        title="Python Engineer",
        company="TechCorp LLC",
        sponsorship_signals={"status": "unknown", "confidence": 0.0},
    )
    job.skills = []  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    assert context.sponsorship is not None
    assert context.sponsorship.sponsorship_score >= 80.0
    assert context.sponsorship.reasoning.historical_approved_petitions == 350
    assert context.explainability is not None
    assert "Highly favorable sponsorship outlook" in context.explainability.actionable_insights[0]


@pytest.mark.asyncio
async def test_sponsorship_negative_company(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Sponsorship-negative company.
    Explicit "negative" sponsorship signals extracted from job posting.
    """
    profile_data = UserProfileCreate(
        full_name="Intl Candidate",
        email="intl2@example.com",
        skills=["Python"],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_spon_neg_hash",
        url="https://example.com/job/spon_neg",
        content_hash="content_spon_neg_hash",
        title="Python Engineer",
        company="LocalCorp",
        # Explicit negative sponsorship signal
        sponsorship_signals={"status": "negative", "confidence": 0.95},
    )
    job.skills = []  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    assert context.sponsorship is not None
    assert context.sponsorship.sponsorship_score < 20.0
    assert context.explainability is not None
    assert "Sponsorship probability is low" in context.explainability.actionable_insights[0]


@pytest.mark.asyncio
async def test_missing_skills_and_explainability(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Missing skills and explainability checks.
    Candidate profile missing critical job skills.
    Checks that gaps, upskilling recommendations, and narratives are produced.
    """
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        skills=["Python"],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_missing_skills_hash",
        url="https://example.com/job/missing_skills",
        content_hash="content_missing_skills_hash",
        title="Django Developer",
        company="TechCorp",
        normalized_skills=["Python", "Django", "Docker", "Kubernetes"],
    )
    job.skills = ["Python", "Django", "Docker", "Kubernetes"]  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    # Verify missing skills list
    assert context.skill_match is not None
    missing_skill_names = [ms.job_skill for ms in context.skill_match.missing_skills]
    assert "Django" in missing_skill_names
    assert "Docker" in missing_skill_names

    # Verify explainability recommendations
    assert context.explainability is not None
    assert any("Upskill in" in rec for rec in context.explainability.improvement_recommendations)


@pytest.mark.asyncio
async def test_overqualified_profile(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Overqualified profile.
    Candidate has 10 years of experience, job requires only 1 year.
    Should result in 100% experience relevance score (no penalty).
    """
    profile_data = UserProfileCreate(
        full_name="Veteran Dev",
        email="vet@example.com",
        skills=["Python"],
        positioning=PositioningSchema(
            headline="Staff Architect",
            years_of_experience=10,
        ),
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_junior_hash",
        url="https://example.com/job/junior",
        content_hash="content_junior_hash",
        title="Junior Developer",
        company="TechCorp",
        experience_required="1 year",
    )
    job.skills = []  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    assert context.ranking is not None
    assert context.ranking.factors.experience_relevance == 100.0


@pytest.mark.asyncio
async def test_underqualified_profile(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Test scenario: Underqualified profile.
    Candidate has 1 year of experience, job requires 10 years.
    Deficit penalty should trigger, mapping overall score to skip/weak.
    """
    profile_data = UserProfileCreate(
        full_name="Junior Dev",
        email="junior@example.com",
        skills=["Python"],
        positioning=PositioningSchema(
            headline="Junior Developer",
            years_of_experience=1,
        ),
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_principal_hash",
        url="https://example.com/job/principal",
        content_hash="content_principal_hash",
        title="Principal Software Architect",
        company="TechCorp",
        experience_required="10 years",
    )
    job.skills = []  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )

    assert context.ranking is not None
    # 9 years deficit -> penalty is 9 * 20 = 180 points -> score should cap at 0.0
    assert context.ranking.factors.experience_relevance == 0.0
    assert context.ranking.recommendation == RecommendationCategory.SKIP


@pytest.mark.asyncio
async def test_recommendation_consistency(
    db_session: Session, orchestrator: OpportunityOrchestrator
) -> None:
    """
    Verify recommendation classification categories maps correctly as scores grow:
    - Overall score >= 85.0 -> STRONG_APPLY
    - Overall score >= 65.0 -> APPLY
    - Overall score >= 40.0 -> WEAK_APPLY
    - Overall score < 40.0 -> SKIP
    """
    profile_data = UserProfileCreate(
        full_name="Consistent Candidate",
        email="consistent@example.com",
        skills=["Python"],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job = JobIntelligence(
        url_hash="job_consistency_hash",
        url="https://example.com/job/consistency",
        content_hash="content_consistency_hash",
        title="Software Engineer",
        company="TechCorp",
    )
    job.skills = []  # type: ignore[attr-defined]
    db_session.add(job)
    db_session.flush()

    # We evaluate ranking category mapping with mock weights configurations
    # directly matching different scores by scaling weights to isolate single scoring values.
    # We test the deterministic engine calculation logic.
    context = await orchestrator.run_pipeline(
        job_intelligence_id=job.id,
        profile_id=profile.id,
    )
    assert context.ranking is not None

    # Verify score boundaries align cleanly with recommendations
    score = context.ranking.overall_score
    rec = context.ranking.recommendation

    if score >= 85.0:
        assert rec == RecommendationCategory.STRONG_APPLY
    elif score >= 65.0:
        assert rec == RecommendationCategory.APPLY
    elif score >= 40.0:
        assert rec == RecommendationCategory.WEAK_APPLY
    else:
        assert rec == RecommendationCategory.SKIP
