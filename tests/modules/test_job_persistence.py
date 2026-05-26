from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.database.models import JobIntelligence
from src.modules.job_ingestion import (
    JobIntelligenceRepository,
    JobPersistenceService,
    compute_content_hash,
    compute_url_hash,
)
from src.modules.matching import (
    DetectionResult,
    SignalType,
    SponsorshipSignal,
    SponsorshipStatus,
)
from src.modules.scraping.schemas import (
    EmploymentType,
    JobDomain,
    JobIntelligenceSchema,
    VisaSignal,
)


@pytest.fixture
def mock_extracted_data() -> JobIntelligenceSchema:
    return JobIntelligenceSchema(
        company="Acme Corp",
        title="Senior Python Developer",
        skills=["Python", "SQL", "Docker"],
        experience_required="5+ years",
        location="Remote",
        visa_signal=VisaSignal.SPONSORSHIP_AVAILABLE,
        employment_type=EmploymentType.FULL_TIME,
        domain=JobDomain.SOFTWARE_ENGINEERING,
        confidence_score=0.9,
    )


@pytest.fixture
def mock_sponsorship_result() -> DetectionResult:
    return DetectionResult(
        status=SponsorshipStatus.POSITIVE,
        confidence=0.85,
        explanation="Offers H1B sponsorship mentioned in text.",
        signals=[
            SponsorshipSignal(
                signal_type=SignalType.SPONSORSHIP_MENTION,
                matched_text="visa sponsorship available",
                score=0.9,
                is_positive=True,
            )
        ],
    )


def test_url_and_content_hashing() -> None:
    # URL hash
    url = "https://example.com/job/1"
    h1 = compute_url_hash(url)
    h2 = compute_url_hash(url + " ")
    assert h1 == h2  # whitespace is stripped

    # Fallback url hash
    h3 = compute_url_hash(None, company="Acme", title="Dev")
    h4 = compute_url_hash("", company=" Acme", title="Dev ")
    assert h3 == h4

    # Content hash
    c1 = compute_content_hash(" hello world ")
    c2 = compute_content_hash("hello world")
    assert c1 == c2


def test_repository_basic_operations(db_session: Session) -> None:
    repo = JobIntelligenceRepository(db_session)

    # 1. Create a record
    url = "https://example.com/jobs/python-dev"
    url_hash = compute_url_hash(url)
    content = "Looking for a python dev with SQL experience"
    content_hash = compute_content_hash(content)

    record = JobIntelligence(
        url_hash=url_hash,
        url=url,
        content_hash=content_hash,
        raw_content=content,
        title="Python Developer",
        company="Global Tech",
        location="New York",
        normalized_skills=["Python", "SQL"],
        domain="software_engineering",
        employment_type="full_time",
    )

    repo.create(record)
    db_session.commit()

    # 2. Get by URL and url_hash
    fetched_by_hash = repo.get_by_url_hash(url_hash)
    assert fetched_by_hash is not None
    assert fetched_by_hash.company == "Global Tech"

    fetched_by_url = repo.get_by_url(url)
    assert fetched_by_url is not None
    assert fetched_by_url.id == record.id

    # 3. Search by skill
    python_matches = repo.search_by_skill("Python")
    assert len(python_matches) == 1
    assert python_matches[0].company == "Global Tech"

    # Test exact vs substring matching on skill search
    sql_matches = repo.search_by_skill("SQL")
    assert len(sql_matches) == 1

    nosql_matches = repo.search_by_skill("NoSQL")
    assert len(nosql_matches) == 0

    # 4. Search by company (case-insensitive keyword matching)
    company_matches = repo.search_by_company("global")
    assert len(company_matches) == 1
    assert company_matches[0].title == "Python Developer"

    # Delete
    repo.delete(record.id)
    db_session.commit()
    assert repo.get_by_url_hash(url_hash) is None


def test_unique_url_hash_constraint(db_session: Session) -> None:
    repo = JobIntelligenceRepository(db_session)
    url_hash = compute_url_hash("https://example.com/unique-job")

    rec1 = JobIntelligence(
        url_hash=url_hash,
        content_hash="hash1",
        normalized_skills=[],
    )
    repo.create(rec1)
    db_session.flush()

    rec2 = JobIntelligence(
        url_hash=url_hash,
        content_hash="hash2",
        normalized_skills=[],
    )
    with pytest.raises(IntegrityError):
        repo.create(rec2)

    db_session.rollback()


def test_service_persist_job_sync_deduplication(
    db_session: Session,
    mock_extracted_data: JobIntelligenceSchema,
    mock_sponsorship_result: DetectionResult,
) -> None:
    service = JobPersistenceService(db_session)
    raw_content = "This is a job posting description for Acme Corp."
    url = "https://acme.com/jobs/1"

    # 1. First persistence call (creates new record)
    record1 = service.persist_job_sync(
        raw_content=raw_content,
        url=url,
        extracted_data=mock_extracted_data,
        sponsorship_result=mock_sponsorship_result,
        ats_type="Greenhouse",
    )
    db_session.commit()
    assert record1.id is not None
    assert record1.url_hash == compute_url_hash(url)

    # 2. Second persistence call with identical content (deduplicates)
    record2 = service.persist_job_sync(
        raw_content=raw_content,
        url=url,
        extracted_data=mock_extracted_data,
        sponsorship_result=mock_sponsorship_result,
        ats_type="Greenhouse",
    )
    assert record1.id == record2.id

    # Verify no additional record was created
    all_records = db_session.query(JobIntelligence).all()
    assert len(all_records) == 1


def test_service_persist_job_sync_update_detection(
    db_session: Session,
    mock_extracted_data: JobIntelligenceSchema,
    mock_sponsorship_result: DetectionResult,
) -> None:
    service = JobPersistenceService(db_session)
    url = "https://acme.com/jobs/1"

    # 1. First persistence call
    record1 = service.persist_job_sync(
        raw_content="Initial description text.",
        url=url,
        extracted_data=mock_extracted_data,
        sponsorship_result=mock_sponsorship_result,
        ats_type="Greenhouse",
    )
    db_session.commit()
    initial_content_hash = record1.content_hash

    # Wait a brief moment to ensure updated_at changes if modified
    # We can manually set updated_at backward to test the modification detection safely
    record1.updated_at = datetime.now(UTC).replace(tzinfo=None) - pytest.importorskip(
        "datetime"
    ).timedelta(seconds=5)
    db_session.commit()
    old_updated_at = record1.updated_at

    # 2. Update with modified content (should trigger update detection)
    modified_content = "Initial description text with updated details."
    updated_extracted_data = mock_extracted_data.model_copy()
    updated_extracted_data.title = "Lead Python Architect"
    updated_extracted_data.skills = ["Python", "SQL", "Docker", "AWS"]

    record2 = service.persist_job_sync(
        raw_content=modified_content,
        url=url,
        extracted_data=updated_extracted_data,
        sponsorship_result=mock_sponsorship_result,
        ats_type="Greenhouse",
    )
    db_session.commit()

    # Verify same DB row but updated content, skills, title, and bumped updated_at
    assert record1.id == record2.id
    assert record2.title == "Lead Python Architect"
    assert "AWS" in record2.normalized_skills
    assert record2.content_hash != initial_content_hash
    assert record2.updated_at > old_updated_at


def test_service_autodetect_fallbacks(
    db_session: Session,
    mock_extracted_data: JobIntelligenceSchema,
) -> None:
    service = JobPersistenceService(db_session)

    # GreenHouse URL and sponsorship terms in content
    url = "https://boards.greenhouse.io/acme/jobs/1"
    raw_content = (
        "Must be authorized to work. We are unable to sponsor visa applications at this time."
    )

    # Persist with sponsorship_result=None and ats_type=None
    record = service.persist_job_sync(
        raw_content=raw_content,
        url=url,
        extracted_data=mock_extracted_data,
        sponsorship_result=None,
        ats_type=None,
    )
    db_session.commit()

    # 1. Verify ATS was automatically detected as greenhouse
    assert record.ats_type == "greenhouse"

    # 2. Verify Sponsorship was automatically scanned and detected as negative
    assert record.sponsorship_signals is not None
    assert record.sponsorship_signals["status"] == "negative"


@pytest.mark.asyncio
async def test_service_async_persist_job(
    db_session: Session,
    mock_extracted_data: JobIntelligenceSchema,
    mock_sponsorship_result: DetectionResult,
) -> None:
    service = JobPersistenceService(db_session)
    url = "https://acme.com/jobs/async-1"
    raw_content = "Async job posting content details here."

    # Run the async persist wrapper
    record = await service.persist_job(
        raw_content=raw_content,
        url=url,
        extracted_data=mock_extracted_data,
        sponsorship_result=mock_sponsorship_result,
        ats_type="Lever",
    )

    # Run a simple check on the returned model
    assert record.id is not None
    assert record.url_hash == compute_url_hash(url)
    assert record.ats_type == "Lever"
