from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.modules.relationship.contacts import ContactService
from src.modules.relationship.models import OutreachEventModel
from src.modules.relationship.repositories import (
    ContactRepository,
    FollowupRepository,
    OutreachEventRepository,
    RelationshipMemoryRepository,
)
from src.modules.relationship.schemas import (
    ContactResponse,
    ContactUpdate,
    InteractionOutcome,
    OutreachEvent,
    OutreachStatus,
    RelationshipMemory,
    RelationshipStatus,
)


class InteractionTimelineEvent(BaseModel):
    """Represents a structured timeline event for contact interactions."""
    event_type: str = Field(description="Type of event (e.g., outreach, response, state_change)")
    timestamp: datetime = Field(description="Timestamp of the event occurrence")
    description: str = Field(description="Concise description of the event")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured event metadata details")


class InteractionRecencyEvaluator:
    """Evaluates relationship recency metrics and staseness."""

    @staticmethod
    def get_last_interaction_date(
        outreach_events: list[OutreachEvent], memory: RelationshipMemory | None
    ) -> datetime | None:
        """Resolves the most recent interaction timestamp from events and memory."""
        last_date: datetime | None = None

        if memory and memory.last_interaction_date:
            last_date = memory.last_interaction_date

        for e in outreach_events:
            event_date = e.completed_at or e.scheduled_for or e.created_at
            if event_date:
                if last_date is None or event_date > last_date:
                    last_date = event_date

        return last_date

    @staticmethod
    def is_stale(last_interaction_date: datetime | None, threshold_days: int = 30) -> bool:
        """Determines if a relationship is stale based on last interaction date and threshold."""
        if not last_interaction_date:
            return True
        limit = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=threshold_days)
        return last_interaction_date.replace(tzinfo=None) < limit


class RelationshipStateManager:
    """Deterministic relationship state transition coordinator."""

    @staticmethod
    def _is_followup_pending(followup_dates: list[datetime] | None, outreach_events: list[OutreachEvent]) -> bool:
        if not followup_dates:
            return False
        now_ts = datetime.now(UTC).replace(tzinfo=None)
        for f_date in followup_dates:
            if f_date.replace(tzinfo=None) <= now_ts:
                addressed = False
                for e in outreach_events:
                    e_date = e.completed_at or e.created_at
                    if e_date and e_date.replace(tzinfo=None) >= f_date.replace(tzinfo=None):
                        addressed = True
                        break
                if not addressed:
                    return True
        return False

    @staticmethod
    def evaluate_state(
        contact: ContactResponse,
        outreach_events: list[OutreachEvent],
        memory: RelationshipMemory | None = None,
        followup_dates: list[datetime] | None = None,
        threshold_days: int = 30,
    ) -> RelationshipStatus:
        """
        Deterministically evaluates the state of a contact.
        Order of evaluation precedence:
        1. Keep DORMANT/ARCHIVED as is.
        2. Evaluate STALE if last interaction is > threshold_days ago.
        3. Evaluate FOLLOWUP_PENDING if a followup suggested date is in the past/today and has not been addressed.
        4. Evaluate ACTIVE if there is an active outcome or a positive reply.
        5. Evaluate RESPONDED if there is any response/reply (neutral/negative).
        6. Evaluate CONTACTED if an outreach has been sent but no response.
        7. Default is NEW.
        """
        # 1. Archive & Dormant check
        if contact.status in {RelationshipStatus.ARCHIVED, RelationshipStatus.DORMANT}:
            return contact.status

        # Resolve last interaction date
        last_interaction = InteractionRecencyEvaluator.get_last_interaction_date(outreach_events, memory)

        # 2. Stale check
        if last_interaction and InteractionRecencyEvaluator.is_stale(last_interaction, threshold_days):
            return RelationshipStatus.STALE

        # 3. Followup Pending check
        if RelationshipStateManager._is_followup_pending(followup_dates, outreach_events):
            return RelationshipStatus.FOLLOWUP_PENDING

        # Filter completed outreach events
        completed_events = [e for e in outreach_events if e.completed_at is not None]
        if not completed_events:
            return RelationshipStatus.NEW

        # Sort completed events by date descending to find the latest
        completed_events.sort(
            key=lambda e: (e.completed_at or e.created_at).replace(tzinfo=None), reverse=True
        )
        latest_event = completed_events[0]

        # 4 & 5. Active & Responded checks
        has_replied = any(e.status == OutreachStatus.REPLIED for e in completed_events)
        has_positive = any(e.outcome == InteractionOutcome.POSITIVE for e in completed_events)

        if has_replied or latest_event.outcome is not None:
            if has_positive or latest_event.outcome == InteractionOutcome.POSITIVE:
                return RelationshipStatus.ACTIVE
            return RelationshipStatus.RESPONDED

        # 6. Contacted check
        return RelationshipStatus.CONTACTED


class InteractionTimelineBuilder:
    """Generates structured, chronological history log of events for a relationship."""

    @staticmethod
    def build_timeline(
        contact: ContactResponse,
        outreach_events: list[OutreachEvent],
        memory: RelationshipMemory | None = None,
    ) -> list[InteractionTimelineEvent]:
        """Compile and sort events chronologically."""
        events: list[InteractionTimelineEvent] = []

        # 1. Contact Creation Event
        events.append(
            InteractionTimelineEvent(
                event_type="state_change",
                timestamp=contact.created_at,
                description="Contact created",
                metadata={"status": RelationshipStatus.NEW.value},
            )
        )

        # 2. Outreach and Response events
        for oe in outreach_events:
            # Add outreach attempt event
            outreach_time = oe.created_at
            events.append(
                InteractionTimelineEvent(
                    event_type="outreach",
                    timestamp=outreach_time,
                    description=f"Outreach attempt via {oe.method.title()}",
                    metadata={
                        "event_id": str(oe.id),
                        "method": oe.method,
                        "status": oe.status.value,
                        "content": oe.content,
                    },
                )
            )

            # If completed and replied, add response event
            if oe.completed_at and oe.status == OutreachStatus.REPLIED:
                events.append(
                    InteractionTimelineEvent(
                        event_type="response",
                        timestamp=oe.completed_at,
                        description=f"Response received via {oe.method.title()}",
                        metadata={
                            "event_id": str(oe.id),
                            "method": oe.method,
                            "outcome": oe.outcome.value if oe.outcome else None,
                        },
                    )
                )

        # 3. Memory notes (if any)
        if memory and memory.last_interaction_date:
            events.append(
                InteractionTimelineEvent(
                    event_type="state_change",
                    timestamp=memory.last_interaction_date,
                    description="Last memory state snapshot update",
                    metadata={"key_facts": memory.key_facts},
                )
            )

        # Sort all events chronologically
        events.sort(key=lambda x: x.timestamp.replace(tzinfo=None))
        return events


class OutreachTrackingService:
    """Orchestrates structured event tracking, responses logging, and timeline generation."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.contact_service = ContactService(session)
        self.contact_repo = ContactRepository(session)
        self.event_repo = OutreachEventRepository(session)
        self.memory_repo = RelationshipMemoryRepository(session)
        self.followup_repo = FollowupRepository(session)

    def _sync_relationship_state(self, contact_id: UUID, threshold_days: int = 30) -> RelationshipStatus:
        """Calculates and updates the persisted relationship status for a contact."""
        contact = self.contact_repo.get_contact_by_uuid(contact_id)
        if not contact:
            raise ValueError(f"Contact with ID {contact_id} not found")

        events = list(self.event_repo.get_events_for_contact(contact_id))
        memory = self.memory_repo.get_memory(contact_id)

        # Fetch followups for contact
        followups = self.followup_repo.get_pending_followups(limit=100)
        contact_followup_dates = [f.suggested_date for f in followups if f.contact_id == contact_id]

        new_status = RelationshipStateManager.evaluate_state(
            contact=contact,
            outreach_events=events,
            memory=memory,
            followup_dates=contact_followup_dates,
            threshold_days=threshold_days,
        )

        if new_status != contact.status:
            self.contact_repo.update_contact(contact_id, ContactUpdate(status=new_status))
            self.session.flush()

        return new_status

    def log_outreach_event(
        self,
        contact_id: UUID,
        method: str,
        content: str | None = None,
        scheduled_for: datetime | None = None,
        completed_at: datetime | None = None,
        outcome: InteractionOutcome | None = None,
    ) -> OutreachEvent:
        """Logs a new outreach attempt event and triggers state progression."""
        status = OutreachStatus.SENT if completed_at else OutreachStatus.PLANNED

        event_schema = OutreachEvent(
            id=uuid4(),
            contact_id=contact_id,
            status=status,
            method=method,
            content=content,
            scheduled_for=scheduled_for,
            completed_at=completed_at,
            outcome=outcome,
            created_at=datetime.now(UTC),
        )

        logged_event = self.event_repo.create_event(event_schema)
        self.session.flush()

        # Update relationship state
        self._sync_relationship_state(contact_id)

        return logged_event

    def log_response(
        self,
        contact_id: UUID,
        method: str,
        content: str | None = None,
        outcome: InteractionOutcome = InteractionOutcome.NEUTRAL,
    ) -> OutreachEvent | None:
        """Logs a response received from the contact, transitioning state to RESPONDED or ACTIVE."""
        # Find the latest pending/sent outreach event to associate the response to
        events = list(self.event_repo.get_events_for_contact(contact_id))
        if not events:
            # If no events exist, log a completed outreach event with REPLIED status directly
            return self.log_outreach_event(
                contact_id=contact_id,
                method=method,
                content=content,
                completed_at=datetime.now(UTC),
                outcome=outcome,
            )

        # Sort events by date descending to find latest
        events.sort(key=lambda e: e.created_at.replace(tzinfo=None), reverse=True)
        latest_event = events[0]

        # Update latest event to REPLIED and record outcome and completion date
        # Since repository update is not explicitly defined in OutreachEventRepository,
        # we update the database model directly
        model = self.session.query(OutreachEventModel).filter_by(id=str(latest_event.id)).first()
        if model:
            model.status = OutreachStatus.REPLIED.value
            model.completed_at = datetime.now(UTC)
            model.outcome = outcome.value
            if content:
                model.content = (model.content or "") + f"\n[Response]: {content}"
            self.session.flush()

            # Update relationship state
            self._sync_relationship_state(contact_id)

            return OutreachEvent.model_validate(
                {**model.__dict__, "id": UUID(model.id), "contact_id": UUID(model.contact_id)}
            )

        return None

    def get_timeline(self, contact_id: UUID) -> list[InteractionTimelineEvent]:
        """Builds chronological interaction timeline for a relationship."""
        contact = self.contact_repo.get_contact_by_uuid(contact_id)
        if not contact:
            raise ValueError(f"Contact with ID {contact_id} not found")

        events = list(self.event_repo.get_events_for_contact(contact_id))
        memory = self.memory_repo.get_memory(contact_id)

        return InteractionTimelineBuilder.build_timeline(
            contact=contact, outreach_events=events, memory=memory
        )

    def get_stale_contacts(self, threshold_days: int = 30) -> list[ContactResponse]:
        """Fetch all contacts that are classified as STALE."""
        contacts = self.contact_repo.list_contacts(limit=1000)
        stale_contacts: list[ContactResponse] = []

        for c in contacts:
            # Sync first to ensure DB state matches current conditions
            current_status = self._sync_relationship_state(c.id, threshold_days=threshold_days)
            if current_status == RelationshipStatus.STALE:
                # Refetch to get the updated status
                updated_c = self.contact_repo.get_contact_by_uuid(c.id)
                if updated_c:
                    stale_contacts.append(updated_c)

        return stale_contacts

    def get_pending_followup_contacts(self) -> list[ContactResponse]:
        """Fetch all contacts that are classified as FOLLOWUP_PENDING."""
        contacts = self.contact_repo.list_contacts(limit=1000)
        pending_contacts: list[ContactResponse] = []

        for c in contacts:
            # Sync first to ensure DB state matches current conditions
            current_status = self._sync_relationship_state(c.id)
            if current_status == RelationshipStatus.FOLLOWUP_PENDING:
                # Refetch to get the updated status
                updated_c = self.contact_repo.get_contact_by_uuid(c.id)
                if updated_c:
                    pending_contacts.append(updated_c)

        return pending_contacts
