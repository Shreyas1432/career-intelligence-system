from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from src.modules.relationship import (
    ContactNormalizer,
    ContactService,
    RelationshipScoreBreakdown,
    RelationshipScorer,
)
from src.modules.relationship.schemas import (
    CommunicationProfile,
    ContactCreate,
    ContactResponse,
    ContactType,
    ContactUpdate,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    RelationshipStatus,
)


def test_contact_normalization_behavior() -> None:
    """Validate name, email, and company normalization logic."""
    # Test Name Normalization
    assert ContactNormalizer.normalize_name("  john doe  ") == "John Doe"
    assert ContactNormalizer.normalize_name("JANE") == "Jane"
    assert ContactNormalizer.normalize_name("") == ""
    assert ContactNormalizer.normalize_name(None) == ""

    # Test Email Normalization
    assert ContactNormalizer.normalize_email("  JOHN@example.com  ") == "john@example.com"
    assert ContactNormalizer.normalize_email("") == ""
    assert ContactNormalizer.normalize_email(None) == ""

    # Test Company Normalization
    assert ContactNormalizer.normalize_company("Google LLC") == "google"
    assert ContactNormalizer.normalize_company("Stripe, Inc.") == "stripe"
    assert ContactNormalizer.normalize_company("Amazon Web Services Ltd.") == "amazon web"
    assert ContactNormalizer.normalize_company("") == ""
    assert ContactNormalizer.normalize_company(None) == ""


def test_enrich_company_from_email() -> None:
    """Validate lightweight email domain company name enrichment."""
    # If company exists, preserve it
    assert ContactService.enrich_company_from_email("test@google.com", "My Company") == "My Company"
    # Personal email domain should return None
    assert ContactService.enrich_company_from_email("test@gmail.com", None) is None
    # Corporate email domain should enrich company name
    assert ContactService.enrich_company_from_email("john@stripe.com", None) == "Stripe"
    # Invalid email returns None
    assert ContactService.enrich_company_from_email("invalid-email", None) is None
    assert ContactService.enrich_company_from_email(None, None) is None


def test_contact_creation_and_duplicate_detection(db_session: Session) -> None:
    """Validate duplicate detection rules on contact creation."""
    service = ContactService(db_session)

    create_data = ContactCreate(
        first_name="Alice",
        last_name="Smith",
        company="Acme Corporation",
        title="Recruiting Coordinator",
        contact_type=ContactType.RECRUITER,
        linkedin_url="https://linkedin.com/in/alicesmith",
        email="alice@acme.com",
    )

    # 1. Create successfully
    contact = service.create_contact(create_data)
    assert isinstance(contact.id, UUID)
    assert contact.first_name == "Alice"
    assert contact.last_name == "Smith"
    assert contact.company == "Acme"  # Normalized "Acme Corporation" -> "Acme" via normalizer
    assert contact.email == "alice@acme.com"

    # 2. Try creating duplicate (same email) - should fail
    dup_email = ContactCreate(
        first_name="Alice Duplicate",
        email="alice@acme.com",
    )
    with pytest.raises(ValueError, match="Duplicate contact detected"):
        service.create_contact(dup_email)

    # Allow creating duplicate if flag is disabled
    c_allowed = service.create_contact(dup_email, raise_on_duplicate=False)
    assert c_allowed.first_name == "Alice Duplicate"

    # 3. Try creating duplicate by LinkedIn (ignoring https/www)
    dup_li = ContactCreate(
        first_name="Alice Li Dup",
        linkedin_url="http://www.linkedin.com/in/alicesmith/",
    )
    with pytest.raises(ValueError, match="Duplicate contact detected"):
        service.create_contact(dup_li)

    # 4. Try creating duplicate by name + company matching
    dup_name_comp = ContactCreate(
        first_name="Alice",
        last_name="Smith",
        company="Acme Corp",  # Normalized to "acme", matching first contact
    )
    with pytest.raises(ValueError, match="Duplicate contact detected"):
        service.create_contact(dup_name_comp)


def test_relationship_status_updates(db_session: Session) -> None:
    """Validate contact status updates and field normalization on update."""
    service = ContactService(db_session)

    contact = service.create_contact(
        ContactCreate(first_name="Bob", last_name="Jones", company="Beta Inc")
    )

    # Update status and check normalization of names/emails
    update_data = ContactUpdate(
        first_name="Robert",
        company="Beta Solutions LLC",
        email="  BOB@beta.com  ",
        status=RelationshipStatus.ACTIVE,
    )
    updated = service.update_contact(contact.id, update_data)
    assert updated is not None
    assert updated.first_name == "Robert"
    assert updated.company == "Beta"  # Normalized Beta Solutions LLC -> Beta
    assert updated.email == "bob@beta.com"  # Normalized email
    assert updated.status == RelationshipStatus.ACTIVE


def test_deterministic_scoring_logic() -> None:
    """Validate that scoring returns deterministic breakdown and correct scores."""
    # 1. Base Recruiter contact with no extra history or target matches
    contact = ContactResponse(
        id=uuid4(),
        first_name="Jane",
        contact_type=ContactType.RECRUITER,
        company="Google",
        status=RelationshipStatus.NEW,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    score = RelationshipScorer.calculate_score(
        contact=contact,
        outreach_events=[],
        memory=None,
        comm_profile=None,
        target_companies=["Google"],
    )

    # Recruiter = 20 pts
    # Google is target company = 20 pts
    # NEW status = 10 pts
    # No recency = 0 pts
    # No engagement = 0 pts
    # Expected total = 50.0
    assert isinstance(score, RelationshipScoreBreakdown)
    assert score.total_score == 50.0
    assert score.company_relevance == 20.0
    assert score.role_score == 20.0
    assert score.status_score == 10.0
    assert score.recency_score == 0.0
    assert score.engagement_score == 0.0

    # 2. Hiring Manager contact with active interactions and target company
    contact_hm = ContactResponse(
        id=uuid4(),
        first_name="Mark",
        contact_type=ContactType.HIRING_MANAGER,
        company="Apple",
        status=RelationshipStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    now_ts = datetime.now(UTC)
    events = [
        OutreachEvent(
            contact_id=contact_hm.id,
            status=OutreachStatus.SENT,
            method="Email",
            outcome=InteractionOutcome.POSITIVE,
            completed_at=now_ts - timedelta(days=5),
        ),
        OutreachEvent(
            contact_id=contact_hm.id,
            status=OutreachStatus.SENT,
            method="LinkedIn",
            outcome=InteractionOutcome.POSITIVE,
            completed_at=now_ts - timedelta(days=10),
        ),
    ]

    profile = CommunicationProfile(
        contact_id=contact_hm.id,
        engagement_score=0.8,
    )

    score_hm = RelationshipScorer.calculate_score(
        contact=contact_hm,
        outreach_events=events,
        memory=None,
        comm_profile=profile,
        target_companies=["Apple Inc"],
    )

    # Hiring Manager = 30 pts
    # Apple matches target "Apple Inc" = 20 pts
    # ACTIVE status = 25 pts
    # Last interaction is 5 days ago (<= 14 days) = 15 pts
    # Profile engagement score (0.8 * 10 = 8 pts) + 2 positive outcomes (2 * 2 = 4 pts) -> capped at 10.0 = 10 pts
    # Expected total = 30 + 20 + 25 + 15 + 10 = 100.0
    assert score_hm.total_score == 100.0
    assert score_hm.company_relevance == 20.0
    assert score_hm.role_score == 30.0
    assert score_hm.status_score == 25.0
    assert score_hm.recency_score == 15.0
    assert score_hm.engagement_score == 10.0
    assert "matches target companies" in score_hm.explanation
    assert "Hiring manager role" in score_hm.explanation


def test_recruiter_prioritization_and_filtering(db_session: Session) -> None:
    """Validate that list and prioritization filters work correctly."""
    service = ContactService(db_session)

    service.create_contact(
        ContactCreate(first_name="Tom", company="Google", contact_type=ContactType.RECRUITER)
    )
    service.create_contact(
        ContactCreate(first_name="Jerry", company="Stripe", contact_type=ContactType.HIRING_MANAGER)
    )

    # List all
    all_contacts = service.list_contacts()
    assert len(all_contacts) >= 2

    # Filter by contact_type
    recruiters = service.list_contacts(contact_type=ContactType.RECRUITER)
    assert any(c.first_name == "Tom" for c in recruiters)
    assert not any(c.first_name == "Jerry" for c in recruiters)

    # Filter by company
    stripe_contacts = service.list_contacts(company="Stripe Inc")
    assert any(c.first_name == "Jerry" for c in stripe_contacts)
    assert not any(c.first_name == "Tom" for c in stripe_contacts)


def test_stable_sorting_behavior(db_session: Session) -> None:
    """Validate that prioritizing preserves relative database order for items with identical scores."""
    service = ContactService(db_session)

    # Create contacts with the exact same scoring parameters (same contact_type, company, status)
    service.create_contact(
        ContactCreate(first_name="Zack", company="Acme Corp", contact_type=ContactType.RECRUITER)
    )
    service.create_contact(
        ContactCreate(first_name="Aaron", company="Acme Corp", contact_type=ContactType.RECRUITER)
    )
    service.create_contact(
        ContactCreate(first_name="Charlie", company="Acme Corp", contact_type=ContactType.RECRUITER)
    )

    # Retrieve all contacts in database order
    all_contacts = service.list_contacts(limit=100)
    db_ordered_names = [
        c.first_name for c in all_contacts if c.first_name in {"Zack", "Aaron", "Charlie"}
    ]

    # Prioritize relationships
    prioritized = service.prioritize_relationships(target_companies=["Acme Corp"])
    prioritized_names = [
        item[0].first_name for item in prioritized if item[0].first_name in {"Zack", "Aaron", "Charlie"}
    ]

    # Because their scores are identical, the stable sort must preserve the database order exactly
    assert prioritized_names == db_ordered_names
