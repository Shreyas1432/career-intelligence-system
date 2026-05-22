from sqlalchemy.orm import Session

from src.core.database.models import UserProfile

from .base import BaseRepository


class UserProfileRepository(BaseRepository[UserProfile]):
    """
    Data repository for UserProfile operations.
    """

    def __init__(self, session: Session):
        super().__init__(UserProfile, session)

    def get_by_email(self, email: str) -> UserProfile | None:
        """
        Lookup user profile card by email.
        """
        return self.session.query(UserProfile).filter(UserProfile.email == email).first()

    def get_active_profile(self) -> UserProfile | None:
        """
        Retrieves the primary active profile (assumes single-user setup, fetching the first row).
        """
        return self.session.query(UserProfile).first()
