import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.ats_detector import detect_ats_sync
from src.modules.job_extraction.schemas import JobIntelligenceSchema
from src.modules.job_persistence.repository import (
    JobIntelligenceRepository,
    compute_content_hash,
    compute_url_hash,
)
from src.modules.sponsorship import DetectionResult, scan_rules

logger = structlog.get_logger("src.modules.job_persistence.service")


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
