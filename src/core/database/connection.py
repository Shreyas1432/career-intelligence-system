from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import PROJECT_ROOT, settings

# Ensure data directory exists
db_dir = PROJECT_ROOT / "data"
db_dir.mkdir(parents=True, exist_ok=True)

# Database Engine Configuration
DATABASE_URL = settings.database.url

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_recycle=settings.database.pool_recycle,
    echo=settings.database.echo_sql,
)


# SQLite Optimizations (WAL mode + normal sync) for M5 MacBook Air performance
@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


# Session builder
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base class from models to ensure a unified registry
from src.core.database.models import Base  # noqa: E402


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions, ensuring sessions are closed cleanly.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Creates all tables in the database if they do not already exist.
    """
    Base.metadata.create_all(bind=engine)
