from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from src.core.database.repositories.base import BaseRepository
from src.modules.relationship.models import (
    CommunicationProfileModel,
    FollowupRecommendationModel,
    OutreachEventModel,
    RelationshipContactModel,
    RelationshipMemoryModel,
)
from src.modules.relationship.schemas import (
    CommunicationProfile,
    CommunicationStyle,
    ContactCreate,
    ContactResponse,
    ContactType,
    ContactUpdate,
    FollowupRecommendation,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    RelationshipMemory,
    RelationshipStatus,
)


class ContactRepository(BaseRepository[RelationshipContactModel]):
    """Repository handling core Contact mapping."""

    def __init__(self, session: Session) -> None:
        super().__init__(RelationshipContactModel, session)

    def _to_schema(self, model: RelationshipContactModel) -> ContactResponse:
        return ContactResponse(
            id=UUID(model.id),
            first_name=model.first_name,
            last_name=model.last_name,
            company=model.company,
            title=model.title,
            contact_type=ContactType(model.contact_type),
            linkedin_url=model.linkedin_url,
            email=model.email,
            status=RelationshipStatus(model.status),
            metadata=model.metadata_json or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def create_contact(self, create_schema: ContactCreate) -> ContactResponse:
        """Create a new contact."""
        model = RelationshipContactModel(
            first_name=create_schema.first_name,
            last_name=create_schema.last_name,
            company=create_schema.company,
            title=create_schema.title,
            contact_type=create_schema.contact_type.value,
            linkedin_url=create_schema.linkedin_url,
            email=create_schema.email,
            status=RelationshipStatus.NEW.value,
            metadata_json=create_schema.metadata,
        )
        self.session.add(model)
        self.session.flush()
        return self._to_schema(model)

    def get_contact_by_uuid(self, contact_id: UUID) -> ContactResponse | None:
        """Retrieve a contact."""
        model = self.session.query(RelationshipContactModel).filter_by(id=str(contact_id)).first()
        return self._to_schema(model) if model else None

    def update_contact(self, contact_id: UUID, update_schema: ContactUpdate) -> ContactResponse | None:
        """Update a contact."""
        model = self.session.query(RelationshipContactModel).filter_by(id=str(contact_id)).first()
        if not model:
            return None

        update_data = update_schema.model_dump(exclude_unset=True)
        if "metadata" in update_data:
            model.metadata_json = update_data.pop("metadata")
        if "contact_type" in update_data:
            model.contact_type = update_data.pop("contact_type")
        if "status" in update_data:
            model.status = update_data.pop("status")

        for key, value in update_data.items():
            setattr(model, key, value)

        self.session.flush()
        return self._to_schema(model)

    def list_contacts(self, skip: int = 0, limit: int = 100, status: RelationshipStatus | None = None) -> Sequence[ContactResponse]:
        """Fetch list of contacts."""
        query = self.session.query(RelationshipContactModel)
        if status:
            query = query.filter_by(status=status.value)
        models = query.offset(skip).limit(limit).all()
        return [self._to_schema(m) for m in models]


class OutreachEventRepository(BaseRepository[OutreachEventModel]):
    """Repository handling outreach events mapping."""

    def __init__(self, session: Session) -> None:
        super().__init__(OutreachEventModel, session)

    def _to_schema(self, model: OutreachEventModel) -> OutreachEvent:
        return OutreachEvent(
            id=UUID(model.id),
            contact_id=UUID(model.contact_id),
            status=OutreachStatus(model.status),
            method=model.method,
            content=model.content,
            scheduled_for=model.scheduled_for,
            completed_at=model.completed_at,
            outcome=InteractionOutcome(model.outcome) if model.outcome else None,
            created_at=model.created_at,
        )

    def create_event(self, event: OutreachEvent) -> OutreachEvent:
        """Logs a new interaction."""
        model = OutreachEventModel(
            id=str(event.id),
            contact_id=str(event.contact_id),
            status=event.status.value,
            method=event.method,
            content=event.content,
            scheduled_for=event.scheduled_for,
            completed_at=event.completed_at,
            outcome=event.outcome.value if event.outcome else None,
            created_at=event.created_at,
        )
        self.session.add(model)
        self.session.flush()
        return self._to_schema(model)

    def get_events_for_contact(self, contact_id: UUID) -> Sequence[OutreachEvent]:
        """Get all outreach items for a given contact."""
        models = self.session.query(OutreachEventModel).filter_by(contact_id=str(contact_id)).all()
        return [self._to_schema(m) for m in models]


class RelationshipMemoryRepository(BaseRepository[RelationshipMemoryModel]):
    """Repository handling context memory."""

    def __init__(self, session: Session) -> None:
        super().__init__(RelationshipMemoryModel, session)

    def _to_schema(self, model: RelationshipMemoryModel) -> RelationshipMemory:
        return RelationshipMemory(
            id=UUID(model.id),
            contact_id=UUID(model.contact_id),
            key_facts=model.key_facts or [],
            shared_interests=model.shared_interests or [],
            pain_points=model.pain_points or [],
            last_interaction_date=model.last_interaction_date,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def get_memory(self, contact_id: UUID) -> RelationshipMemory | None:
        """Fetch relationship context."""
        model = self.session.query(RelationshipMemoryModel).filter_by(contact_id=str(contact_id)).first()
        return self._to_schema(model) if model else None

    def save_memory(self, memory: RelationshipMemory) -> RelationshipMemory:
        """Persist memory."""
        model = self.session.query(RelationshipMemoryModel).filter_by(contact_id=str(memory.contact_id)).first()
        if not model:
            model = RelationshipMemoryModel(
                id=str(memory.id),
                contact_id=str(memory.contact_id),
            )
            self.session.add(model)

        model.key_facts = memory.key_facts
        model.shared_interests = memory.shared_interests
        model.pain_points = memory.pain_points
        model.last_interaction_date = memory.last_interaction_date

        self.session.flush()
        return self._to_schema(model)


class CommunicationProfileRepository(BaseRepository[CommunicationProfileModel]):
    """Repository handling communication styles."""

    def __init__(self, session: Session) -> None:
        super().__init__(CommunicationProfileModel, session)

    def _to_schema(self, model: CommunicationProfileModel) -> CommunicationProfile:
        return CommunicationProfile(
            contact_id=UUID(model.contact_id),
            style=CommunicationStyle(model.style),
            best_time_to_reach=model.best_time_to_reach,
            preferred_channel=model.preferred_channel,
            engagement_score=model.engagement_score,
            insights=model.insights or [],
        )

    def get_profile(self, contact_id: UUID) -> CommunicationProfile | None:
        """Get the profile config."""
        model = self.session.query(CommunicationProfileModel).filter_by(contact_id=str(contact_id)).first()
        return self._to_schema(model) if model else None

    def update_profile(self, profile: CommunicationProfile) -> CommunicationProfile:
        """Persist profile."""
        model = self.session.query(CommunicationProfileModel).filter_by(contact_id=str(profile.contact_id)).first()
        if not model:
            model = CommunicationProfileModel(contact_id=str(profile.contact_id))
            self.session.add(model)

        model.style = profile.style.value
        model.best_time_to_reach = profile.best_time_to_reach
        model.preferred_channel = profile.preferred_channel
        model.engagement_score = profile.engagement_score
        model.insights = profile.insights

        self.session.flush()
        return self._to_schema(model)


class FollowupRepository(BaseRepository[FollowupRecommendationModel]):
    """Repository handing follow-up task recommendations tracking."""

    def __init__(self, session: Session) -> None:
        super().__init__(FollowupRecommendationModel, session)

    def _to_schema(self, model: FollowupRecommendationModel) -> FollowupRecommendation:
        return FollowupRecommendation(
            contact_id=UUID(model.contact_id),
            suggested_date=model.suggested_date,
            reasoning=model.reasoning,
            draft_message=model.draft_message,
            priority=model.priority,
        )

    def get_pending_followups(self, limit: int = 50) -> Sequence[FollowupRecommendation]:
        """Fetch unresolved recommended follow-ups."""
        models = self.session.query(FollowupRecommendationModel).limit(limit).all()
        return [self._to_schema(m) for m in models]
