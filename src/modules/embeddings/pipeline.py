import asyncio
from typing import Any

from src.modules.embeddings.cache import EmbeddingCache
from src.modules.embeddings.service import EmbeddingService


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
