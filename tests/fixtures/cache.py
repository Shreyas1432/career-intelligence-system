from collections.abc import Generator
from pathlib import Path

import pytest

from src.core.cache import CacheManager, SQLiteCacheStore
from src.core.config import PROJECT_ROOT


@pytest.fixture
def temp_cache_db() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary SQLite cache database path.
    """
    test_db = PROJECT_ROOT / "data" / "test_cache.db"
    if test_db.exists():
        test_db.unlink()

    yield test_db

    if test_db.exists():
        try:
            test_db.unlink()
        except OSError:
            pass


@pytest.fixture
def test_cache_store(temp_cache_db: Path) -> Generator[SQLiteCacheStore, None, None]:
    """
    Fixture providing a clean SQLiteCacheStore instance.
    """
    store = SQLiteCacheStore(db_path=temp_cache_db)
    yield store
    store.close()


@pytest.fixture
def test_cache_manager(test_cache_store: SQLiteCacheStore) -> CacheManager:
    """
    Fixture providing a CacheManager using the test store.
    """
    return CacheManager(store=test_cache_store)
