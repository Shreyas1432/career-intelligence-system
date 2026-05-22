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
        Fetch application eagerly loading job metadata.
        """
        return self.session.query(Application).filter(Application.id == application_id).first()


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
