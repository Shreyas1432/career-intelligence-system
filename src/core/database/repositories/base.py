from typing import Generic, TypeVar

from sqlalchemy.orm import Session

from src.core.database.models import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):  # noqa: UP046
    """
    Generic Base Repository pattern for general database CRUD operations.
    """

    def __init__(self, model_class: type[T], session: Session):
        self.model_class = model_class
        self.session = session

    def get_by_id(self, id_val: int) -> T | None:
        """
        Fetch a single entity by its primary key ID.
        """
        return (
            self.session.query(self.model_class)
            .filter(self.model_class.id == id_val)  # type: ignore[attr-defined]
            .first()
        )

    def get_all(self, skip: int = 0, limit: int = 100) -> list[T]:
        """
        Fetch lists of entities with offset pagination.
        """
        return self.session.query(self.model_class).offset(skip).limit(limit).all()

    def create(self, entity: T) -> T:
        """
        Persists a new entity instance into the session.
        """
        self.session.add(entity)
        self.session.flush()  # Populates auto-generated ID without committing transaction
        return entity

    def delete(self, id_val: int) -> bool:
        """
        Remove entity instance by its ID. Returns True if removed, False otherwise.
        """
        obj = self.get_by_id(id_val)
        if obj:
            self.session.delete(obj)
            self.session.flush()
            return True
        return False
