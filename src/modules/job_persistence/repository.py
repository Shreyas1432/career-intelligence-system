import hashlib

from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.sql import column

from src.core.database.models import JobIntelligence
from src.core.database.repositories.base import BaseRepository
from src.modules.skill_normalization import canonicalize_skills


def compute_url_hash(url: str | None, company: str | None = None, title: str | None = None) -> str:
    """
    Compute a unique SHA-256 hash for a job posting.
    If url is provided and not empty, hashes the URL.
    Otherwise, hashes 'company:title'.
    """
    if url and url.strip():
        return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()

    # Fallback to company:title
    comp = (company or "").strip()
    tit = (title or "").strip()
    fallback_str = f"{comp}:{tit}"
    return hashlib.sha256(fallback_str.encode("utf-8")).hexdigest()


def compute_content_hash(content: str | None) -> str:
    """
    Compute SHA-256 hash of the raw job posting text.
    """
    return hashlib.sha256((content or "").strip().encode("utf-8")).hexdigest()


class JobIntelligenceRepository(BaseRepository[JobIntelligence]):
    """
    Data repository for JobIntelligence persistence and retrieval operations.
    """

    def __init__(self, session: Session):
        super().__init__(JobIntelligence, session)

    def get_by_url_hash(self, url_hash: str) -> JobIntelligence | None:
        """
        Fetch a job intelligence record by its unique URL hash.
        """
        if not url_hash:
            return None
        return (
            self.session.query(JobIntelligence).filter(JobIntelligence.url_hash == url_hash).first()
        )

    def get_by_url(self, url: str) -> JobIntelligence | None:
        """
        Fetch a job intelligence record by its URL.
        Hashes the URL to query by the indexed url_hash field.
        """
        if not url or not url.strip():
            return None
        url_hash = compute_url_hash(url)
        return self.get_by_url_hash(url_hash)

    def search_by_skill(self, skill: str) -> list[JobIntelligence]:
        """
        Search job intelligence records containing a specific skill in their normalized skills.
        Uses SQLite json_each function for exact JSON list containment.
        """
        if not skill or not skill.strip():
            return []

        normalized = canonicalize_skills([skill])
        canonical_skill = normalized[0] if normalized else skill.strip()

        return (
            self.session.query(JobIntelligence)
            .filter(
                self.session.query(1)
                .select_from(func.json_each(JobIntelligence.normalized_skills))
                .filter(column("value") == canonical_skill)
                .exists()
            )
            .all()
        )

    def search_by_company(self, company: str) -> list[JobIntelligence]:
        """
        Search job intelligence records by company name (case-insensitive keyword matching).
        """
        if not company or not company.strip():
            return []
        search_pattern = f"%{company.strip()}%"
        return (
            self.session.query(JobIntelligence)
            .filter(JobIntelligence.company.ilike(search_pattern))
            .all()
        )
