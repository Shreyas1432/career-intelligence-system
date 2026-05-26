from uuid import UUID

from sqlalchemy.orm import Session

from src.modules.relationship.repositories import (
    CommunicationProfileRepository,
    ContactRepository,
    OutreachEventRepository,
    RelationshipMemoryRepository,
)
from src.modules.relationship.schemas import (
    CommunicationProfile,
    CommunicationStyle,
    ContactCreate,
    ContactType,
    ContactUpdate,
    OutreachEvent,
    OutreachStatus,
    RelationshipMemory,
    RelationshipStatus,
)


def test_contact_crud(db_session: Session) -> None:
    repo = ContactRepository(db_session)

    # 1. Create
    create_schema = ContactCreate(
        first_name="Jane",
        last_name="Doe",
        company="Acme Corp",
        title="Software Engineer",
        contact_type=ContactType.PEER,
        email="jane@example.com",
    )
    contact = repo.create_contact(create_schema)
    assert isinstance(contact.id, UUID)
    assert contact.first_name == "Jane"
    assert contact.contact_type == ContactType.PEER
    assert contact.status == RelationshipStatus.NEW

    # 2. Read UUID
    fetched = repo.get_contact_by_uuid(contact.id)
    assert fetched is not None
    assert fetched.first_name == "Jane"

    # 3. Update behavior
    update_schema = ContactUpdate(status=RelationshipStatus.WARM, title="Senior Engineer")
    updated = repo.update_contact(contact.id, update_schema)
    assert updated is not None
    assert updated.status == RelationshipStatus.WARM
    assert updated.title == "Senior Engineer"

    # Verify update state holds
    fetched_again = repo.get_contact_by_uuid(contact.id)
    assert fetched_again is not None
    assert fetched_again.status == RelationshipStatus.WARM


def test_filtered_retrieval(db_session: Session) -> None:
    repo = ContactRepository(db_session)

    contacts = [
        ContactCreate(first_name=f"User{i}") for i in range(5)
    ]
    for c in contacts:
        repo.create_contact(c)

    all_contacts = repo.list_contacts(limit=10)
    assert len(all_contacts) >= 5

    # Filtering check
    new_contacts = repo.list_contacts(status=RelationshipStatus.NEW)
    assert len(new_contacts) >= 5


def test_outreach_events_persistence(db_session: Session) -> None:
    contact_repo = ContactRepository(db_session)
    event_repo = OutreachEventRepository(db_session)

    contact = contact_repo.create_contact(ContactCreate(first_name="Tom"))

    event = OutreachEvent(
        contact_id=contact.id,
        status=OutreachStatus.SENT,
        method="LinkedIn",
        content="Hello!",
    )

    created = event_repo.create_event(event)
    assert created.status == OutreachStatus.SENT

    events = event_repo.get_events_for_contact(contact.id)
    assert len(events) == 1
    assert events[0].method == "LinkedIn"


def test_relationship_memory(db_session: Session) -> None:
    contact_repo = ContactRepository(db_session)
    repo = RelationshipMemoryRepository(db_session)

    contact = contact_repo.create_contact(ContactCreate(first_name="Alice"))

    memory = RelationshipMemory(
        contact_id=contact.id,
        key_facts=["Likes python", "Knows Rust"],
    )
    saved = repo.save_memory(memory)
    assert "Likes python" in saved.key_facts

    fetched = repo.get_memory(contact.id)
    assert fetched is not None
    assert "Knows Rust" in fetched.key_facts

    # Validate Optional / Defaults handling
    assert fetched.last_interaction_date is None


def test_communication_profile(db_session: Session) -> None:
    contact_repo = ContactRepository(db_session)
    repo = CommunicationProfileRepository(db_session)

    contact = contact_repo.create_contact(ContactCreate(first_name="Bob"))

    profile = CommunicationProfile(
        contact_id=contact.id,
        style=CommunicationStyle.DIRECT,
        preferred_channel="Slack",
        engagement_score=0.8,
    )

    updated = repo.update_profile(profile)
    assert updated.engagement_score == 0.8

    fetched = repo.get_profile(contact.id)
    assert fetched is not None
    assert fetched.style == CommunicationStyle.DIRECT
