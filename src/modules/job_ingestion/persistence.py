import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.sql import column

from src.core.database.models import JobIntelligence
from src.core.database.repositories.base import BaseRepository
from src.modules.matching.sponsorship import DetectionResult, scan_rules
from src.modules.scraping.ats_detection import detect_ats_sync
from src.modules.scraping.normalization import canonicalize_skills
from src.modules.scraping.schemas import JobIntelligenceSchema

logger = structlog.get_logger("src.modules.job_ingestion")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

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


def serialize_sponsorship_result(res: DetectionResult | None) -> dict[str, Any] | None:
    """
    Serialize a DetectionResult dataclass to a database-friendly JSON dict.
    """
    if res is None:
        return None
    return {
        "status": res.status.value if hasattr(res.status, "value") else str(res.status),
        "confidence": res.confidence,
        "explanation": res.explanation,
        "signals": [
            {
                "signal_type": (
                    sig.signal_type.value
                    if hasattr(sig.signal_type, "value")
                    else str(sig.signal_type)
                ),
                "matched_text": sig.matched_text,
                "score": sig.score,
                "is_positive": sig.is_positive,
            }
            for sig in res.signals
        ],
    }


# ------------------------------------------------------------------------------
# Repository
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Service
# ------------------------------------------------------------------------------

class JobPersistenceService:
    """
    Service layer coordinating job persistence, deduplication, update detection,
    and automatic signal enhancements.
    """

    def __init__(self, session: Session):
        self.session = session
        self.repository = JobIntelligenceRepository(session)

    def persist_job_sync(
        self,
        raw_content: str,
        url: str | None,
        extracted_data: JobIntelligenceSchema,
        sponsorship_result: DetectionResult | None = None,
        ats_type: str | None = None,
    ) -> JobIntelligence:
        """
        Synchronously persist or update job intelligence in the database.
        Checks for existing postings via url_hash and checks for changes via content_hash.
        """
        # 1. Compute hashes
        url_hash = compute_url_hash(url, extracted_data.company, extracted_data.title)
        content_hash = compute_content_hash(raw_content)

        # 2. Check if record exists
        existing = self.repository.get_by_url_hash(url_hash)

        # 3. Handle sponsorship result fallback
        if sponsorship_result is None:
            logger.debug("Sponsorship result not provided; scanning rules")
            sponsorship_result = scan_rules(raw_content or "")

        serialized_sponsorship = serialize_sponsorship_result(sponsorship_result)

        # 4. Handle ATS type fallback
        if not ats_type:
            logger.debug("ATS type not provided; running ATS detector")
            ats = detect_ats_sync(url or "", html=raw_content)
            ats_type = ats.value if hasattr(ats, "value") else str(ats)

        # Naive UTC datetime for standard SQLAlchemy timezone-naive columns
        now_utc = datetime.now(UTC).replace(tzinfo=None)

        if existing:
            # Check content hash for updates
            if existing.content_hash == content_hash:
                logger.info(
                    "Job posting already exists with identical content; skipping write (deduplication)",
                    url_hash=url_hash,
                )
                return existing

            # Hashes differ, update existing record
            logger.info(
                "Job posting exists but content has changed; updating record (update detection)",
                url_hash=url_hash,
            )
            existing.raw_content = raw_content
            existing.content_hash = content_hash
            existing.title = extracted_data.title
            existing.company = extracted_data.company
            existing.location = extracted_data.location
            existing.experience_required = extracted_data.experience_required
            existing.domain = (
                extracted_data.domain.value
                if hasattr(extracted_data.domain, "value")
                else str(extracted_data.domain)
            )
            existing.employment_type = (
                extracted_data.employment_type.value
                if hasattr(extracted_data.employment_type, "value")
                else str(extracted_data.employment_type)
            )
            existing.confidence_score = extracted_data.confidence_score
            existing.normalized_skills = extracted_data.skills
            existing.ats_type = ats_type
            existing.sponsorship_signals = serialized_sponsorship
            existing.updated_at = now_utc

            self.session.flush()
            return existing

        # Create new record
        logger.info(
            "Persisting new job intelligence record",
            url_hash=url_hash,
        )
        new_record = JobIntelligence(
            url_hash=url_hash,
            url=url,
            content_hash=content_hash,
            raw_content=raw_content,
            title=extracted_data.title,
            company=extracted_data.company,
            location=extracted_data.location,
            experience_required=extracted_data.experience_required,
            domain=(
                extracted_data.domain.value
                if hasattr(extracted_data.domain, "value")
                else str(extracted_data.domain)
            ),
            employment_type=(
                extracted_data.employment_type.value
                if hasattr(extracted_data.employment_type, "value")
                else str(extracted_data.employment_type)
            ),
            confidence_score=extracted_data.confidence_score,
            normalized_skills=extracted_data.skills,
            ats_type=ats_type,
            sponsorship_signals=serialized_sponsorship,
            created_at=now_utc,
            updated_at=now_utc,
        )

        self.repository.create(new_record)
        return new_record

    async def persist_job(
        self,
        raw_content: str,
        url: str | None,
        extracted_data: JobIntelligenceSchema,
        sponsorship_result: DetectionResult | None = None,
        ats_type: str | None = None,
    ) -> JobIntelligence:
        """
        Asynchronously persist or update job intelligence in the database.
        Offloads synchronous database operation to a worker thread.
        """
        return await asyncio.to_thread(
            self.persist_job_sync,
            raw_content=raw_content,
            url=url,
            extracted_data=extracted_data,
            sponsorship_result=sponsorship_result,
            ats_type=ats_type,
        )
