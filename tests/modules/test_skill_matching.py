import numpy as np
import pytest

from src.modules.matching.skill_matching import (
    MatchType,
    SkillMatchingEngine,
    calculate_domain_alignment_bonus,
    calculate_procurement_bonus,
    calculate_skill_weight,
)
from src.modules.scraping.schemas import JobDomain, SkillCategory


class MockSentenceTransformerForMatching:
    """
    Mock SentenceTransformer model that returns deterministic unit vectors
    yielding specific cosine similarity values for controlled test cases.
    - Similarity("AWS", "Azure") = 0.85
    - Similarity("Docker", "Kubernetes") = 0.80
    - Others = close to 0.0
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
            elif "docker" in t:
                vec[2] = 1.0
            elif "kubernetes" in t:
                vec[2] = 0.80
                vec[3] = 0.60
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
    Automatically mock SentenceTransformer with MockSentenceTransformerForMatching.
    """
    mock_model = MockSentenceTransformerForMatching()
    monkeypatch.setattr(
        "src.modules.matching.embeddings.SentenceTransformer", lambda _name: mock_model
    )
    return mock_model


@pytest.fixture
def matching_engine():
    return SkillMatchingEngine()


def test_calculate_skill_weight():
    # Test base weights without domain multiplier
    w_prog_other = calculate_skill_weight(SkillCategory.PROGRAMMING, JobDomain.OTHER)
    assert w_prog_other == 1.2

    # Test weight with domain multiplier (SOFTWARE_ENGINEERING multiplies PROGRAMMING by 1.5)
    w_prog_se = calculate_skill_weight(SkillCategory.PROGRAMMING, JobDomain.SOFTWARE_ENGINEERING)
    assert w_prog_se == 1.8  # 1.2 * 1.5

    # Test weight with domain multiplier (OPERATIONS multiplies SUPPLY_CHAIN by 1.8)
    w_sc_ops = calculate_skill_weight(SkillCategory.SUPPLY_CHAIN, JobDomain.OPERATIONS)
    assert w_sc_ops == 2.16  # 1.2 * 1.8 = 2.16


def test_calculate_procurement_bonus():
    # Only active if job domain is OPERATIONS
    bonus_se = calculate_procurement_bonus(
        JobDomain.SOFTWARE_ENGINEERING, [SkillCategory.PROCUREMENT, SkillCategory.SUPPLY_CHAIN]
    )
    assert bonus_se == 0.0

    # Test incremental scaling (2.5 per match, cap at 10.0)
    bonus_ops_1 = calculate_procurement_bonus(JobDomain.OPERATIONS, [SkillCategory.PROCUREMENT])
    assert bonus_ops_1 == 2.5

    bonus_ops_3 = calculate_procurement_bonus(
        JobDomain.OPERATIONS,
        [SkillCategory.PROCUREMENT, SkillCategory.SUPPLY_CHAIN, SkillCategory.PROCUREMENT],
    )
    assert bonus_ops_3 == 7.5

    # Capped at 10.0
    bonus_ops_5 = calculate_procurement_bonus(
        JobDomain.OPERATIONS,
        [
            SkillCategory.PROCUREMENT,
            SkillCategory.SUPPLY_CHAIN,
            SkillCategory.PROCUREMENT,
            SkillCategory.SUPPLY_CHAIN,
            SkillCategory.PROCUREMENT,
        ],
    )
    assert bonus_ops_5 == 10.0


def test_calculate_domain_alignment_bonus():
    # No matches
    bonus_none = calculate_domain_alignment_bonus(
        JobDomain.SOFTWARE_ENGINEERING, ["data_ai"], "Data Engineer", ["Healthcare"]
    )
    assert bonus_none == 0.0

    # Domain match only (+5.0)
    bonus_domain = calculate_domain_alignment_bonus(
        JobDomain.SOFTWARE_ENGINEERING,
        ["software_engineering"],
        "Staff Engineer",
        ["Healthcare"],
    )
    assert bonus_domain == 5.0

    # Domain + target industry overlap (+10.0)
    bonus_both = calculate_domain_alignment_bonus(
        JobDomain.SOFTWARE_ENGINEERING,
        ["software_engineering"],
        "Healthcare Systems Engineer",
        ["Healthcare"],
    )
    assert bonus_both == 10.0


@pytest.mark.asyncio
async def test_engine_exact_skill_matching(matching_engine):
    user_profile = {
        "skills": ["Python", "SQL"],
        "domains": ["software_engineering"],
        "target_industries": [],
    }
    job_intel = {
        "title": "Python Developer",
        "domain": JobDomain.SOFTWARE_ENGINEERING,
        "skills": ["Python"],
    }

    result = await matching_engine.match_profile_to_job(user_profile, job_intel)

    # Python is an exact match (similarity = 1.0)
    assert (
        result.final_score == 105.0 or result.final_score == 100.0
    )  # Normalized 100% + 5% domain bonus, capped at 100%
    assert result.final_score == 100.0
    assert len(result.matched_skills) == 1
    assert result.matched_skills[0].matched_skill == "Python"
    assert result.matched_skills[0].match_type == MatchType.EXACT
    assert result.matched_skills[0].similarity == 1.0
    assert len(result.missing_skills) == 0


@pytest.mark.asyncio
async def test_engine_semantic_skill_matching(matching_engine):
    # Candidate has Azure. Job requires AWS.
    # From MockSentenceTransformerForMatching: Similarity("AWS", "Azure") = 0.85
    user_profile = {
        "skills": ["Azure"],
        "domains": [],
        "target_industries": [],
    }
    job_intel = {
        "title": "Cloud Infrastructure Engineer",
        "domain": JobDomain.SOFTWARE_ENGINEERING,
        "skills": ["AWS"],
    }

    # Similarity is 0.85, which is above the 0.75 threshold
    result = await matching_engine.match_profile_to_job(
        user_profile, job_intel, similarity_threshold=0.75
    )

    assert len(result.matched_skills) == 1
    match = result.matched_skills[0]
    assert match.job_skill == "AWS"
    assert match.user_skill == "Azure"
    assert match.match_type == MatchType.SEMANTIC
    assert match.similarity == 0.85
    assert len(result.missing_skills) == 0
    # Final score: 85.31% normalized score + 0 bonus = 85.31
    assert result.final_score == 85.31


@pytest.mark.asyncio
async def test_engine_semantic_matching_below_threshold(matching_engine):
    # Candidate has SQL. Job requires AWS.
    # Similarity will be very low (close to 0.0)
    user_profile = {
        "skills": ["SQL"],
        "domains": [],
        "target_industries": [],
    }
    job_intel = {
        "title": "Cloud Infrastructure Engineer",
        "domain": JobDomain.SOFTWARE_ENGINEERING,
        "skills": ["AWS"],
    }

    result = await matching_engine.match_profile_to_job(
        user_profile, job_intel, similarity_threshold=0.75
    )

    assert len(result.matched_skills) == 0
    assert len(result.missing_skills) == 1
    assert result.missing_skills[0].job_skill == "AWS"
    assert result.final_score == 0.0


@pytest.mark.asyncio
async def test_engine_procurement_and_supply_chain_alignment(matching_engine):
    # User has Procurement & Supply Chain skills, job domain is OPERATIONS
    user_profile = {
        "skills": [
            "purchasing",
            "vendor management",
            "forecasting",
        ],  # maps to Procurement, Procurement, Demand Planning
        "domains": ["operations"],
        "target_industries": [],
    }
    job_intel = {
        "title": "Procurement Manager",
        "domain": JobDomain.OPERATIONS,
        "skills": ["Procurement", "Demand Planning", "SQL"],
    }

    # Matching:
    # - Procurement exact match (from "purchasing" / "vendor management")
    # - Demand Planning exact match (from "forecasting")
    # - SQL is missing (since user has purchasing, vendor management, forecasting)
    result = await matching_engine.match_profile_to_job(user_profile, job_intel)

    # Check score breakdown
    assert len(result.matched_skills) == 2
    assert len(result.missing_skills) == 1
    assert (
        result.score_breakdown.domain_alignment_bonus == 5.0
    )  # user domains has "operations", job domain is OPERATIONS
    assert (
        result.score_breakdown.procurement_supply_chain_bonus > 0.0
    )  # Should have matching supply chain/procurement skills bonus
    assert result.final_score > 0.0

    # Ensure explainability report exists and is structured properly
    assert result.explanation.summary != ""
    assert len(result.explanation.strengths) > 0
    assert len(result.explanation.gaps) > 0
    assert len(result.explanation.recommendations) > 0
