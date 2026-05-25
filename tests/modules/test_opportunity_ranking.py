import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.modules.matching import (
    OpportunityRankingEngine,
    RankingWeights,
    RecommendationCategory,
    SponsorshipPersistenceService,
)
from src.modules.positioning.profile import UserProfileCreate, UserProfileService
from src.modules.scraping.schemas import JobDomain


class MockSentenceTransformerForRanking:
    """
    Mock SentenceTransformer model that returns deterministic unit vectors
    yielding specific cosine similarity values for controlled test cases.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dims = 384

    def encode(
        self, texts: str | list[str], _convert_to_numpy: bool = True, **_kwargs
    ) -> np.ndarray:
        single = isinstance(texts, str)
        texts_list = [texts] if single else list(texts)

        results = []
        for text in texts_list:
            t = text.lower().strip()
            vec = np.zeros(self.dims, dtype=np.float32)
            if "aws" in t:
                vec[0] = 1.0
            elif "azure" in t:
                vec[0] = 0.85
                vec[1] = 0.5268
            elif "python" in t:
                vec[4] = 1.0
            elif "sql" in t:
                vec[5] = 1.0
            else:
                # Deterministic random vector based on text hash
                h = hash(text)
                np.random.seed(abs(h) % (2**32))
                random_vec = np.random.randn(self.dims).astype(np.float32)
                random_vec[0:10] = 0.0  # clear controlled indices
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
def mock_sentence_transformer(monkeypatch):
    """
    Automatically mock SentenceTransformer with MockSentenceTransformerForRanking.
    """
    mock_model = MockSentenceTransformerForRanking()
    monkeypatch.setattr(
        "src.modules.matching.embeddings.SentenceTransformer", lambda _name: mock_model
    )
    return mock_model


@pytest.fixture
def ranking_engine():
    return OpportunityRankingEngine()


def test_calculate_experience_relevance(ranking_engine):
    # Candidate positioning: 5 years of experience
    profile = {
        "positioning": {
            "years_of_experience": 5,
            "seniority_level": "Senior",
        }
    }

    # Job: 3 years required (met)
    job_met = {"experience_required": "3 years of experience"}
    assert ranking_engine._calculate_experience_relevance(profile, job_met) == 100.0

    # Job: 5+ years required (met)
    job_met_plus = {"experience_required": "5+ yrs"}
    assert ranking_engine._calculate_experience_relevance(profile, job_met_plus) == 100.0

    # Job: 6 years required (1 year deficient -> 100 - 1*20 = 80)
    job_def1 = {"experience_required": "6 yrs"}
    assert ranking_engine._calculate_experience_relevance(profile, job_def1) == 80.0

    # Job: 10 years required (5 years deficient -> 100 - 5*20 = 0)
    job_def5 = {"experience_required": "10 years required"}
    assert ranking_engine._calculate_experience_relevance(profile, job_def5) == 0.0

    # Seniority fallback (Job has "Senior" in title -> maps to 5 years, met)
    job_sen = {"title": "Senior Software Engineer"}
    assert ranking_engine._calculate_experience_relevance(profile, job_sen) == 100.0


def test_calculate_enterprise_alignment(ranking_engine):
    profile = {
        "target_roles": ["Software Engineer", "Developer"],
        "target_industries": ["Healthcare", "Finance"],
    }

    # Scenario 1: Base (no match) -> 50.0
    job_base = {"title": "Data Analyst", "domain": JobDomain.OPERATIONS}
    assert ranking_engine._calculate_enterprise_alignment(profile, job_base) == 50.0

    # Scenario 2: Title matches target role (+25) -> 75.0
    job_title_match = {"title": "Fullstack Developer", "domain": JobDomain.OPERATIONS}
    assert ranking_engine._calculate_enterprise_alignment(profile, job_title_match) == 75.0

    # Scenario 3: Domain matches target industry (+25) -> 75.0
    job_ind_match = {"title": "Data Analyst", "domain": "Healthcare"}
    assert ranking_engine._calculate_enterprise_alignment(profile, job_ind_match) == 75.0

    # Scenario 4: Both match (+50) -> 100.0
    job_both = {"title": "Healthcare Software Engineer", "domain": "Healthcare"}
    assert ranking_engine._calculate_enterprise_alignment(profile, job_both) == 100.0


@pytest.mark.asyncio
async def test_ranking_engine_avoidance_filter(ranking_engine, db_session: Session):
    # Setup candidate profile in DB
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane@example.com",
        target_roles=["Software Engineer"],
        skills=["Python"],
        experience_summary="Backend dev",
        domains=["software_engineering"],
        target_industries=["Healthcare"],
        avoid_role_filters={
            "avoid_titles": ["Manager"],
            "avoid_companies": ["NastyCorp"],
            "avoid_keywords": ["gambling"],
        },
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    job_avoid = {
        "title": "Engineering Manager",
        "company": "NastyCorp",
        "raw_content": "gambling industry",
        "domain": JobDomain.SOFTWARE_ENGINEERING,
        "skills": ["Python"],
    }

    # Should immediately trigger SKIP with 0.0 score
    resp = await ranking_engine.rank_opportunity(profile, job_avoid, db_session=db_session)
    assert resp.overall_score == 0.0
    assert resp.recommendation == RecommendationCategory.SKIP
    assert "avoidance filters" in resp.reasoning.gaps[0]


@pytest.mark.asyncio
async def test_ranking_engine_weighted_scoring(ranking_engine, db_session: Session):
    # Setup candidate profile
    profile_data = UserProfileCreate(
        full_name="Jane Doe",
        email="jane@example.com",
        target_roles=["Software Engineer"],
        skills=["Python", "SQL"],
        experience_summary="Backend dev",
        domains=["software_engineering"],
        target_industries=["Healthcare"],
    )
    profile = UserProfileService.create_or_update_profile(db_session, profile_data)

    # Setup historical database sponsorship record
    persistence = SponsorshipPersistenceService(db_session)
    persistence.upsert_sponsorship_record("Google LLC", 2024, 100, 2)

    job_intel = {
        "title": "Python Developer",
        "company": "Google LLC",
        "domain": JobDomain.SOFTWARE_ENGINEERING,
        "skills": ["Python"],
        "experience_required": "3 years",
        "sponsorship_signals": {"status": "unknown", "confidence": 0.0},
    }

    # 1. Test under default weights
    resp1 = await ranking_engine.rank_opportunity(profile, job_intel, db_session=db_session)
    assert resp1.overall_score > 0.0
    assert resp1.recommendation in (
        RecommendationCategory.STRONG_APPLY,
        RecommendationCategory.APPLY,
    )
    assert resp1.reasoning.explanation != ""

    # 2. Test under customized weights (focus heavily on sponsorship)
    custom_weights = RankingWeights(
        skill_matching=0.1,
        domain_alignment=0.1,
        sponsorship_probability=0.6,
        experience_relevance=0.1,
        enterprise_alignment=0.1,
    )
    resp2 = await ranking_engine.rank_opportunity(
        profile, job_intel, weights=custom_weights, db_session=db_session
    )
    # Sponsorship score is high (Google H-1B stats are strong)
    # With 60% weight on sponsorship, overall score is heavily dominated by it
    assert resp2.overall_score == 77.75
