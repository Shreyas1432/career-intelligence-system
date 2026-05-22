import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.database.connection import Base


@pytest.fixture(scope="session")
def db_engine():
    """
    Creates an in-memory SQLite database engine for testing.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine):
    """
    Returns an isolated database session for each test case.
    Uses transactional rollback to keep tests fast and isolated.
    """
    connection = db_engine.connect()
    transaction = connection.begin()

    session_class = sessionmaker(bind=connection)
    session = session_class()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
