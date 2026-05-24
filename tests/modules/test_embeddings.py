import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence, UserProfile
from src.modules.embeddings.cache import EmbeddingCache
from src.modules.embeddings.pipeline import (
    EmbeddingPipeline,
    format_job_text,
    format_profile_text,
)
from src.modules.embeddings.repository import EmbeddingRepository
from src.modules.embeddings.service import EmbeddingService


class MockSentenceTransformer:
    """
    Mock SentenceTransformer model that returns deterministic, normalized unit vectors
    of length 384 based on the input text hash, without downloading the real model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.call_count = 0

    def encode(
        self, texts: str | list[str], convert_to_numpy: bool = True, **_kwargs
    ) -> np.ndarray:
        self.call_count += 1
        single = isinstance(texts, str)
        if single:
            texts_list = [texts]
        else:
            texts_list = list(texts)

        results = []
        for text in texts_list:
            # Generate deterministic vector based on text hash
            h = hash(text)
            np.random.seed(abs(h) % (2**32))
            vector = np.random.randn(384).astype(np.float32)
            # Normalize vector to unit length so similarity equals dot product
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            results.append(vector)

        if convert_to_numpy:
            arr = np.array(results)
        else:
            arr = results

        if single:
            return arr[0]
        return arr


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """
    Automatically mock SentenceTransformer for all test cases.
    """
    mock_model = MockSentenceTransformer()
    monkeypatch.setattr(
        "src.modules.embeddings.service.SentenceTransformer", lambda _name: mock_model
    )
    return mock_model


@pytest.fixture(autouse=True)
def clear_embedding_service_model():
    """
    Reset singleton model reference on EmbeddingService before and after each test.
    """
    EmbeddingService.unload_model()
    yield
    EmbeddingService.unload_model()


def test_embedding_service_lifecycle():
    """
    Test that the EmbeddingService lazily loads and unloads the model.
    """
    assert EmbeddingService._model is None

    # Lazy loading
    model1 = EmbeddingService.get_model()
    assert model1 is not None
    assert EmbeddingService._model is model1

    # Second call gets the same instance
    model2 = EmbeddingService.get_model()
    assert model1 is model2

    # Unloading frees resources
    EmbeddingService.unload_model()
    assert EmbeddingService._model is None


def test_embedding_service_generation():
    """
    Test generating embeddings for single text and batch of texts.
    """
    service = EmbeddingService()

    # Single text embedding
    emb = service.generate_embedding_sync("Hello career intelligence")
    assert len(emb) == 384
    assert isinstance(emb, list)
    assert isinstance(emb[0], float)

    # Batch embedding
    embs = service.generate_embeddings_sync(["Hello", "World"])
    assert len(embs) == 2
    assert len(embs[0]) == 384
    assert len(embs[1]) == 384


def test_similarity_calculation():
    """
    Test cosine similarity helper method.
    """
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [1.0, 0.0, 0.0]
    emb3 = [0.0, 1.0, 0.0]

    assert abs(EmbeddingService.calculate_similarity(emb1, emb2) - 1.0) < 1e-6
    assert abs(EmbeddingService.calculate_similarity(emb1, emb3) - 0.0) < 1e-6

    # Verify normalization/cosine similarity maths
    emb4 = [2.0, 0.0, 0.0]
    assert abs(EmbeddingService.calculate_similarity(emb1, emb4) - 1.0) < 1e-6

    # Empty edge cases
    assert EmbeddingService.calculate_similarity([], [1.0, 0.0]) == 0.0


def test_embedding_cache_operations(test_cache_manager):
    """
    Test embedding caching read, write, and batch extraction.
    """
    cache = EmbeddingCache(manager=test_cache_manager)
    text = "cached sample sentence"
    vector = [0.1] * 384

    # Cache miss
    assert cache.get_cached_embedding(text) is None

    # Write cache
    cache.set_cached_embedding(text, vector)

    # Cache hit
    hit = cache.get_cached_embedding(text)
    assert hit == vector

    # Batch retrieval
    texts = [text, "uncached sample"]
    batch_res = cache.get_cached_embeddings_batch(texts)
    assert text in batch_res
    assert batch_res[text] == vector
    assert "uncached sample" not in batch_res


@pytest.mark.asyncio
async def test_embedding_pipeline_batching(test_cache_manager, mock_sentence_transformer):
    """
    Test that the EmbeddingPipeline merges cache hits and encodes cache misses,
    while maintaining the original input order.
    """
    cache = EmbeddingCache(manager=test_cache_manager)
    service = EmbeddingService()
    pipeline = EmbeddingPipeline(service=service, cache=cache)

    # Prime cache for one key
    prime_text = "Already Cached"
    prime_vector = [0.5] * 384
    await cache.set_cached_embedding_async(prime_text, prime_vector)

    texts = [
        "First Miss",
        "Already Cached",
        "Second Miss",
        "Already Cached",
    ]

    # Reset model call count
    mock_sentence_transformer.call_count = 0

    embeddings = await pipeline.embed_texts(texts)

    # Verify result size and matching vectors
    assert len(embeddings) == 4
    assert embeddings[1] == prime_vector
    assert embeddings[3] == prime_vector
    assert len(embeddings[0]) == 384
    assert len(embeddings[2]) == 384

    # Verify that the mock model was called only once for the batch of misses
    # (The batch is ["First Miss", "Second Miss"] because "Already Cached" was cached)
    assert mock_sentence_transformer.call_count == 1

    # Verify the misses are now in the cache
    assert await cache.get_cached_embedding_async("First Miss") == embeddings[0]
    assert await cache.get_cached_embedding_async("Second Miss") == embeddings[2]


def test_text_formatting():
    """
    Test formatting helper functions for profile and job intelligence.
    """
    job_text = format_job_text(
        job_title="Software Engineer",
        company="Tech Corp",
        description="Write Python code",
    )
    assert "Job Title: Software Engineer" in job_text
    assert "Company: Tech Corp" in job_text
    assert "Description: Write Python code" in job_text

    # Profile as dict
    profile_dict = {
        "full_name": "Jane Doe",
        "positioning": {"headline": "AI Specialist", "seniority_level": "Senior"},
        "target_roles": ["ML Engineer", "Researcher"],
        "skills": ["Python", "PyTorch"],
        "experience_summary": "ML researcher at lab",
        "domains": ["AI", "Research"],
        "target_industries": ["Tech", "Bio"],
        "experience": [
            {
                "title": "Research Assistant",
                "company": "Uni Lab",
                "description": "Implemented architectures",
            }
        ],
    }

    profile_text = format_profile_text(profile_dict)
    assert "Headline: AI Specialist" in profile_text
    assert "Seniority: Senior" in profile_text
    assert "Target Roles: ML Engineer, Researcher" in profile_text
    assert "Skills: Python, PyTorch" in profile_text
    assert "Summary: ML researcher at lab" in profile_text
    assert "Domains: AI, Research" in profile_text
    assert "Target Industries: Tech, Bio" in profile_text
    assert "Research Assistant at Uni Lab - Implemented architectures" in profile_text


def test_database_persistence_and_cascade(db_session: Session):
    """
    Test persistence of JobEmbedding and ProfileEmbedding, retrieval,
    and cascading deletion behaviors.
    """
    from sqlalchemy import text

    db_session.execute(text("PRAGMA foreign_keys=ON;"))
    repo = EmbeddingRepository(session=db_session)

    # 1. Setup Parent entities
    job = JobIntelligence(
        url_hash="hash-12345",
        content_hash="content-123",
        title="Solutions Architect",
        company="Global Tech",
        normalized_skills=["AWS", "Python"],
    )
    db_session.add(job)

    profile = UserProfile(
        full_name="Bob Builder",
        email="bob@builder.com",
        skills="Python, SQL",
    )
    db_session.add(profile)
    db_session.flush()

    assert job.id is not None
    assert profile.id is not None

    # 2. Save embeddings
    vector_job = [0.123] * 384
    vector_profile = [0.456] * 384

    repo.save_job_embedding(job.id, vector_job)
    repo.save_profile_embedding(profile.id, vector_profile)

    # 3. Retrieve embeddings
    db_job_emb = repo.get_job_embedding(job.id)
    assert db_job_emb is not None
    assert db_job_emb.embedding == vector_job

    db_profile_emb = repo.get_profile_embedding(profile.id)
    assert db_profile_emb is not None
    assert db_profile_emb.embedding == vector_profile

    # 4. Get all job embeddings (for in-memory search)
    all_job_embs = repo.get_all_job_embeddings()
    assert len(all_job_embs) == 1
    assert all_job_embs[0][0] == job.id
    assert all_job_embs[0][1] == vector_job

    # 5. Verify Cascade Deletes
    # Delete job intelligence -> should remove job embedding
    db_session.delete(job)
    db_session.flush()
    assert repo.get_job_embedding(job.id) is None

    # Delete profile -> should remove profile embedding
    db_session.delete(profile)
    db_session.flush()
    assert repo.get_profile_embedding(profile.id) is None
