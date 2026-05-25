import asyncio
import gc
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from src.core.cache.manager import CacheManager
from src.core.database.models import JobEmbedding, ProfileEmbedding
from src.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingRepository:
    """
    Data repository for Job and User Profile embedding persistence.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_job_embedding(
        self, job_id: int, embedding: list[float], session: Session | None = None
    ) -> JobEmbedding:
        """
        Save or update a job embedding.
        """
        sess = session or self.session
        db_emb = sess.query(JobEmbedding).filter(JobEmbedding.job_id == job_id).first()
        if db_emb:
            db_emb.embedding = embedding
        else:
            db_emb = JobEmbedding(job_id=job_id, embedding=embedding)
            sess.add(db_emb)
        sess.flush()
        return db_emb

    def get_job_embedding(self, job_id: int, session: Session | None = None) -> JobEmbedding | None:
        """
        Retrieve a job embedding by job_id.
        """
        sess = session or self.session
        return sess.query(JobEmbedding).filter(JobEmbedding.job_id == job_id).first()

    def save_profile_embedding(
        self, profile_id: int, embedding: list[float], session: Session | None = None
    ) -> ProfileEmbedding:
        """
        Save or update a profile embedding.
        """
        sess = session or self.session
        db_emb = (
            sess.query(ProfileEmbedding).filter(ProfileEmbedding.profile_id == profile_id).first()
        )
        if db_emb:
            db_emb.embedding = embedding
        else:
            db_emb = ProfileEmbedding(profile_id=profile_id, embedding=embedding)
            sess.add(db_emb)
        sess.flush()
        return db_emb

    def get_profile_embedding(
        self, profile_id: int, session: Session | None = None
    ) -> ProfileEmbedding | None:
        """
        Retrieve a profile embedding by profile_id.
        """
        sess = session or self.session
        return (
            sess.query(ProfileEmbedding).filter(ProfileEmbedding.profile_id == profile_id).first()
        )

    def get_all_job_embeddings(
        self, session: Session | None = None
    ) -> list[tuple[int, list[float]]]:
        """
        Retrieve all job embeddings for in-memory similarity comparisons.
        """
        sess = session or self.session
        records = sess.query(JobEmbedding).all()
        return [(r.job_id, r.embedding) for r in records]


class EmbeddingCache:
    """
    Wrapper around CacheManager specifically for all-MiniLM-L6-v2 embeddings.
    """

    def __init__(self, manager: CacheManager | None = None) -> None:
        self.manager = manager or CacheManager()
        self.model_name = "all-MiniLM-L6-v2"

    def get_cached_embedding(self, text: str) -> list[float] | None:
        """
        Retrieve a cached embedding from SQLite cache store.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        val = self.manager.get(key)
        if val is not None and isinstance(val, list):
            return [float(x) for x in val]
        return None

    def set_cached_embedding(self, text: str, embedding: list[float]) -> None:
        """
        Store a generated embedding in SQLite cache store.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        self.manager.set(key, "embedding", embedding)

    def get_cached_embeddings_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """
        Retrieve cached embeddings for a list of texts in batch.
        """
        results = {}
        for text in texts:
            emb = self.get_cached_embedding(text)
            if emb is not None:
                results[text] = emb
        return results

    # Async variants to prevent blocking the event loop
    async def get_cached_embedding_async(self, text: str) -> list[float] | None:
        """
        Asynchronously retrieve a cached embedding.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        val = await self.manager.get_async(key)
        if val is not None and isinstance(val, list):
            return [float(x) for x in val]
        return None

    async def set_cached_embedding_async(self, text: str, embedding: list[float]) -> None:
        """
        Asynchronously store a generated embedding in cache.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        await self.manager.set_async(key, "embedding", embedding)

    async def get_cached_embeddings_batch_async(self, texts: list[str]) -> dict[str, list[float]]:
        """
        Asynchronously retrieve cached embeddings for a batch of texts.
        """
        results = {}
        for text in texts:
            emb = await self.get_cached_embedding_async(text)
            if emb is not None:
                results[text] = emb
        return results


class EmbeddingService:
    """
    Service for lazy-loading and interacting with sentence-transformer models.
    Optimized for MacBook Air (low RAM footprint, unloadable model).
    """

    _model_name: str = "all-MiniLM-L6-v2"
    _model: Any = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Any:
        """
        Thread-safe lazy retrieval of the sentence transformer model.
        """
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info("Initializing SentenceTransformer model", model=cls._model_name)
                    cls._model = SentenceTransformer(cls._model_name)
        return cls._model

    @classmethod
    def unload_model(cls) -> None:
        """
        Deregister sentence-transformer model and trigger garbage collection to free RAM.
        """
        with cls._lock:
            if cls._model is not None:
                logger.info("Unloading SentenceTransformer model", model=cls._model_name)
                cls._model = None
                gc.collect()

    def generate_embedding_sync(self, text: str) -> list[float]:
        """
        Synchronously generate embedding for a single text.
        """
        if not text:
            return []
        model = self.get_model()
        vector = model.encode(text, convert_to_numpy=True)
        return [float(x) for x in vector.tolist()]

    def generate_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Synchronously generate embeddings for a batch of texts.
        """
        if not texts:
            return []
        model = self.get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in vector] for vector in vectors.tolist()]

    @staticmethod
    def calculate_similarity(emb1: list[float], emb2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Returns a float between -1.0 and 1.0 (or 0.0 for empty vectors).
        """
        if not emb1 or not emb2:
            return 0.0
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_positioning(profile: Any) -> list[str]:
    parts = []
    positioning = _get_val(profile, "positioning")
    if positioning:
        headline = _get_val(positioning, "headline")
        seniority = _get_val(positioning, "seniority_level")
        if headline:
            parts.append(f"Headline: {headline}")
        if seniority:
            parts.append(f"Seniority: {seniority}")
    return parts


def _format_target_roles(profile: Any) -> str | None:
    target_roles = _get_val(profile, "target_roles")
    if target_roles:
        if isinstance(target_roles, str):
            return f"Target Roles: {target_roles}"
        return f"Target Roles: {', '.join(target_roles)}"
    return None


def _format_skills(profile: Any) -> str | None:
    skills = _get_val(profile, "skills")
    if skills:
        if isinstance(skills, str):
            return f"Skills: {skills}"
        return f"Skills: {', '.join(skills)}"
    return None


def _format_experience_items(profile: Any) -> str | None:
    experience = _get_val(profile, "experience")
    if not experience:
        return None
    exp_parts = []
    for exp in experience:
        title = _get_val(exp, "title")
        company = _get_val(exp, "company")
        desc = _get_val(exp, "description")
        if title and company:
            item_str = f"{title} at {company}"
            if desc:
                item_str += f" - {desc}"
            exp_parts.append(item_str)
    if exp_parts:
        return "Work Experience:\n" + "\n".join(exp_parts)
    return None


def format_profile_text(profile: Any) -> str:
    """
    Standardize formatting of a user profile into a single text representation for embeddings.
    """
    parts = []

    parts.extend(_format_positioning(profile))

    roles = _format_target_roles(profile)
    if roles:
        parts.append(roles)

    skills = _format_skills(profile)
    if skills:
        parts.append(skills)

    exp_summary = _get_val(profile, "experience_summary")
    if exp_summary:
        parts.append(f"Summary: {exp_summary}")

    domains = _get_val(profile, "domains")
    if domains:
        parts.append(f"Domains: {', '.join(domains)}")

    industries = _get_val(profile, "target_industries")
    if industries:
        parts.append(f"Target Industries: {', '.join(industries)}")

    exp_items = _format_experience_items(profile)
    if exp_items:
        parts.append(exp_items)

    return "\n".join(parts).strip()


def format_job_text(job_title: str, company: str, description: str) -> str:
    """
    Standardize formatting of a job posting into a single text representation for embeddings.
    """
    parts = []
    if job_title:
        parts.append(f"Job Title: {job_title}")
    if company:
        parts.append(f"Company: {company}")
    if description:
        parts.append(f"Description: {description}")
    return "\n".join(parts).strip()


class EmbeddingPipeline:
    """
    Pipeline for generating embeddings with SQLite caching, async execution,
    and batching support.
    """

    def __init__(
        self, service: EmbeddingService | None = None, cache: EmbeddingCache | None = None
    ):
        self.service = service or EmbeddingService()
        self.cache = cache or EmbeddingCache()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts using batching, caching,
        and thread-pool execution for cache misses.
        """
        if not texts:
            return []

        # Find unique texts to save work
        unique_texts = list(set(texts))

        # Check cache asynchronously
        cached_map = await self.cache.get_cached_embeddings_batch_async(unique_texts)

        # Identify cache misses
        misses = [t for t in unique_texts if t not in cached_map]

        if misses:
            # Generate embeddings for misses in a thread pool
            embeddings_miss = await asyncio.to_thread(self.service.generate_embeddings_sync, misses)

            # Store misses in cache and update cached_map
            for text, emb in zip(misses, embeddings_miss, strict=True):
                cached_map[text] = emb
                await self.cache.set_cached_embedding_async(text, emb)

        # Build results mapping back to the original order
        results = [cached_map[text] for text in texts]
        return results

    async def embed_profile(self, profile: Any) -> list[float]:
        """
        Format a profile and return its embedding.
        """
        text = format_profile_text(profile)
        embs = await self.embed_texts([text])
        return embs[0] if embs else []

    async def embed_job(self, job_title: str, company: str, description: str) -> list[float]:
        """
        Format a job and return its embedding.
        """
        text = format_job_text(job_title, company, description)
        embs = await self.embed_texts([text])
        return embs[0] if embs else []
