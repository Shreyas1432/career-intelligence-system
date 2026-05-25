from sqlalchemy.orm import Session

from src.core.database.models import Application, Contact, InteractionSummary, Job

from .base import BaseRepository


class JobRepository(BaseRepository[Job]):
    """
    Data repository for Job tracking operations.
    """

    def __init__(self, session: Session):
        super().__init__(Job, session)

    def search_jobs(self, query: str) -> list[Job]:
        """
        Search target jobs by title or company keywords.
        """
        search_pattern = f"%{query}%"
        return (
            self.session.query(Job)
            .filter((Job.title.like(search_pattern)) | (Job.company.like(search_pattern)))
            .all()
        )


class ApplicationRepository(BaseRepository[Application]):
    """
    Data repository for Application process stages.
    """

    def __init__(self, session: Session):
        super().__init__(Application, session)

    def get_by_status(self, status: str) -> list[Application]:
        """
        Fetch applications matching a specific milestone status (e.g., Applied, Interviewing).
        """
        return self.session.query(Application).filter(Application.status == status).all()

    def get_application_with_details(self, application_id: int) -> Application | None:
        """
        Fetch application by ID, eagerly loading job metadata.
        Returns None when the application does not exist.
        """
        return self.session.query(Application).filter(Application.id == application_id).first()

    def get_application_with_details_or_raise(self, application_id: int) -> Application:
        """
        Fetch application by ID, eagerly loading job metadata.
        Raises ValueError when the application does not exist.
        Use this variant when the caller holds a known-valid ID (e.g. from a
        preceding get_by_status result) and a missing record is a data-integrity error.
        """
        application = self.session.query(Application).filter(
            Application.id == application_id
        ).first()
        if application is None:
            raise ValueError(f"Application with ID {application_id} not found")
        return application


class ContactRepository(BaseRepository[Contact]):
    """
    Data repository for networking contacts.
    """

    def __init__(self, session: Session):
        super().__init__(Contact, session)

    def get_by_company(self, company: str) -> list[Contact]:
        """
        Find all logged contacts belonging to a targeted organization.
        """
        return self.session.query(Contact).filter(Contact.company == company).all()


class InteractionRepository(BaseRepository[InteractionSummary]):
    """
    Data repository for interaction logging.
    """

    def __init__(self, session: Session):
        super().__init__(InteractionSummary, session)

    def get_recent_interactions(self, limit: int = 5) -> list[InteractionSummary]:
        """
        Fetch most recent logs across application cycles.
        """
        return (
            self.session.query(InteractionSummary)
            .order_by(InteractionSummary.date.desc())
            .limit(limit)
            .all()
        )
