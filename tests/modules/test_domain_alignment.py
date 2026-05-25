import numpy as np
import pytest

from src.modules.matching.domain_alignment import (
    DomainAlignmentEngine,
    DomainCategory,
    clean_text_for_matching,
    extract_matched_keywords,
)
from src.modules.scraping.schemas import JobDomain


class MockSentenceTransformerForAlignment:
    """
    Mock SentenceTransformer model that returns deterministic unit vectors
    yielding specific cosine similarity values for controlled test cases.
    - Similarity("Python developer", "Data Scientist") = 0.80
    - Similarity("SAP consultant", "SAP Integration Specialist") = 0.85
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
            if "sap consultant" in t:
                vec[0] = 1.0
            elif "sap integration specialist" in t:
                vec[0] = 0.85
                vec[1] = 0.5268
            elif "python developer" in t:
                vec[2] = 1.0
            elif "data scientist" in t:
                vec[2] = 0.80
                vec[3] = 0.60
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
    Automatically mock SentenceTransformer with MockSentenceTransformerForAlignment.
    """
    mock_model = MockSentenceTransformerForAlignment()
    monkeypatch.setattr(
        "src.modules.matching.embeddings.SentenceTransformer", lambda _name: mock_model
    )
    return mock_model


@pytest.fixture
def alignment_engine():
    return DomainAlignmentEngine()


def test_clean_text_for_matching():
    assert clean_text_for_matching("SAP ERP Consulting!") == "sap erp consulting"
    assert clean_text_for_matching("Python & Machine-Learning") == "python & machine learning"
    assert clean_text_for_matching("") == ""


def test_extract_matched_keywords():
    kws_enterprise = extract_matched_keywords(
        "SAP ERP Specialist", DomainCategory.ENTERPRISE_SYSTEMS
    )
    assert "sap" in kws_enterprise
    assert "erp" in kws_enterprise

    kws_ai = extract_matched_keywords("Senior ML/AI Engineer", DomainCategory.AI_ANALYTICS)
    assert "ml" in kws_ai
    assert "ai" in kws_ai

    kws_none = extract_matched_keywords("Chef at a restaurant", DomainCategory.PROCUREMENT)
    assert len(kws_none) == 0


def test_resolve_active_domains(alignment_engine):
    # Operations job with SAP and Procurement requirements
    primary = alignment_engine._get_primary_domains(JobDomain.OPERATIONS)
    assert DomainCategory.SUPPLY_CHAIN in primary
    assert DomainCategory.PROCUREMENT in primary

    active = alignment_engine._resolve_active_domains(
        job_title="SAP Logistics Lead",
        job_skills=["SAP", "Strategic Sourcing"],
        primary_domains=primary,
    )
    # SAP maps to enterprise_systems, Strategic Sourcing maps to procurement.
    # OPERATIONS triggers supply_chain and procurement.
    assert DomainCategory.ENTERPRISE_SYSTEMS in active
    assert DomainCategory.PROCUREMENT in active
    assert DomainCategory.SUPPLY_CHAIN in active
    assert DomainCategory.AI_ANALYTICS not in active


@pytest.mark.asyncio
async def test_engine_domain_alignment_strong(alignment_engine):
    user_pos = {
        "headline": "SAP Consultant with Oracle ERP expertise",
        "seniority_level": "Senior",
        "years_of_experience": 8,
    }
    job_intel = {
        "title": "SAP Integration Specialist",
        "domain": JobDomain.OPERATIONS,  # Operations triggers supply_chain, procurement as primary
        "skills": ["SAP"],
    }

    # Similarity between "SAP Consultant..." and "SAP Integration Specialist..." will be 0.85
    result = await alignment_engine.align_domain(user_pos, job_intel)

    assert result.final_score > 0.0
    # SAP / Oracle ERP matches Enterprise Systems (2 keywords) -> active (via job title and skills)
    ent_details = result.domain_breakdown[DomainCategory.ENTERPRISE_SYSTEMS]
    assert "sap" in ent_details.matched_keywords
    assert ent_details.rule_score == 100.0  # 50 + 2 * 25 = 100

    # Semantic similarity check
    assert result.reasoning.semantic_similarity == 0.85
    assert len(result.reasoning.strengths) > 0
    assert result.reasoning.explanation != ""


@pytest.mark.asyncio
async def test_engine_domain_alignment_neutral_transferable(alignment_engine):
    # Candidate positioning is AI/Analytics focused.
    # Job is also AI/Analytics focused.
    user_pos = {
        "headline": "Python Developer and ML engineer",
        "seniority_level": "Junior",
        "years_of_experience": 2,
    }
    job_intel = {
        "title": "Data Scientist",
        "domain": JobDomain.DATA_AI,
        "skills": ["Python", "Machine Learning"],
    }

    # Similarity is 0.80
    result = await alignment_engine.align_domain(user_pos, job_intel)

    assert result.final_score > 70.0
    ai_details = result.domain_breakdown[DomainCategory.AI_ANALYTICS]
    assert "python" in ai_details.matched_keywords
    assert "machine learning" in ai_details.matched_keywords or "ml" in ai_details.matched_keywords
    assert ai_details.rule_score == 100.0

    # Non-active domains like supply_chain should score 100.0 for rule score (since neither candidate claims nor job wants)
    sc_details = result.domain_breakdown[DomainCategory.SUPPLY_CHAIN]
    assert sc_details.rule_score == 100.0
