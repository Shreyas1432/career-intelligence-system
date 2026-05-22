import asyncio
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from src.core.cache import CacheManager, SQLiteCacheStore
from src.core.config import PROJECT_ROOT, settings


@pytest.fixture
def temp_cache_db() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary SQLite cache database path.
    """
    test_db = PROJECT_ROOT / "data" / "test_cache.db"
    # Ensure any residual test db is removed
    if test_db.exists():
        test_db.unlink()

    yield test_db

    # Clean up after test
    if test_db.exists():
        try:
            test_db.unlink()
        except OSError:
            pass


@pytest.fixture
def cache_store(temp_cache_db: Path) -> Generator[SQLiteCacheStore, None, None]:
    """
    Fixture providing a clean SQLiteCacheStore instance.
    """
    store = SQLiteCacheStore(db_path=temp_cache_db)
    yield store
    store.close()


@pytest.fixture
def cache_manager(cache_store: SQLiteCacheStore) -> CacheManager:
    """
    Fixture providing a CacheManager using the test store.
    """
    return CacheManager(store=cache_store)


def test_cache_key_generation(cache_manager: CacheManager) -> None:
    # Embedding key stability
    k1 = cache_manager.generate_embedding_key("model-a", "hello world")
    k2 = cache_manager.generate_embedding_key("model-a", "hello world")
    k3 = cache_manager.generate_embedding_key("model-a", "hello world!")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("embedding:model-a:")

    # AI Response key stability with varying parameter order
    k_ai_1 = cache_manager.generate_ai_response_key(
        model="gpt-4",
        system_prompt="sys",
        prompt="user",
        temperature=0.7,
        max_tokens=100,
        extra_arg=1,
        another_arg="test",
    )
    k_ai_2 = cache_manager.generate_ai_response_key(
        model="gpt-4",
        system_prompt="sys",
        prompt="user",
        temperature=0.7,
        max_tokens=100,
        another_arg="test",  # Different parameter ordering in call signature
        extra_arg=1,
    )
    assert k_ai_1 == k_ai_2
    assert k_ai_1.startswith("ai_response:gpt-4:")


def test_basic_get_set_delete(cache_manager: CacheManager) -> None:
    # Key miss
    assert cache_manager.get("missing_key") is None

    # Key hit after set
    payload = {"choices": [{"text": "Hello text"}], "usage": {"tokens": 12}}
    cache_manager.set("test_key", "response", payload)
    assert cache_manager.get("test_key") == payload

    # Delete key
    assert cache_manager.delete("test_key") is True
    assert cache_manager.get("test_key") is None

    # Delete non-existing key
    assert cache_manager.delete("test_key") is False


def test_ttl_expiration(cache_manager: CacheManager) -> None:
    # Normal active key
    cache_manager.set("active_key", "response", "val", ttl_seconds=300)
    assert cache_manager.get("active_key") == "val"

    # Expired key using negative TTL
    cache_manager.set("expired_key", "response", "expired_val", ttl_seconds=-10)
    # Check that it returns None (and triggers internal deletion)
    assert cache_manager.get("expired_key") is None


def test_delete_by_prefix(cache_manager: CacheManager) -> None:
    cache_manager.set("embedding:model-a:hash1", "embedding", [0.1, 0.2])
    cache_manager.set("embedding:model-a:hash2", "embedding", [0.3, 0.4])
    cache_manager.set("ai_response:gpt-4:hash3", "response", "answer")

    deleted = cache_manager.delete_by_prefix("embedding:")
    assert deleted == 2

    assert cache_manager.get("embedding:model-a:hash1") is None
    assert cache_manager.get("embedding:model-a:hash2") is None
    assert cache_manager.get("ai_response:gpt-4:hash3") == "answer"


def test_cache_disabled_mode(cache_manager: CacheManager, monkeypatch: pytest.MonkeyPatch) -> None:
    # Disable cache globally via monkeypatching setting
    monkeypatch.setattr(settings.cache, "enabled", False)

    cache_manager.set("disabled_key", "response", "val")
    assert cache_manager.get("disabled_key") is None

    # Ensure get_or_set executes creator_fn but does not write cache
    counter = 0

    def creator() -> str:
        nonlocal counter
        counter += 1
        return f"run-{counter}"

    res1 = cache_manager.get_or_set("disabled_key", "response", creator)
    res2 = cache_manager.get_or_set("disabled_key", "response", creator)

    assert res1 == "run-1"
    assert res2 == "run-2"
    assert counter == 2
    assert cache_manager.get("disabled_key") is None


def test_get_or_set(cache_manager: CacheManager) -> None:
    counter = 0

    def creator() -> str:
        nonlocal counter
        counter += 1
        return f"val-{counter}"

    # Hit creator_fn on miss
    res1 = cache_manager.get_or_set("k1", "response", creator)
    assert res1 == "val-1"

    # Re-use cached value on second call
    res2 = cache_manager.get_or_set("k1", "response", creator)
    assert res2 == "val-1"
    assert counter == 1


@pytest.mark.asyncio
async def test_async_operations(cache_manager: CacheManager) -> None:
    await cache_manager.set_async("async_k", "response", "async_val")
    assert await cache_manager.get_async("async_k") == "async_val"

    # Test get_or_set_async with async supplier callable
    counter = 0

    async def async_creator() -> str:
        nonlocal counter
        counter += 1
        await asyncio.sleep(0.01)
        return f"async-{counter}"

    res1 = await cache_manager.get_or_set_async("async_lazy", "response", async_creator)
    res2 = await cache_manager.get_or_set_async("async_lazy", "response", async_creator)

    assert res1 == "async-1"
    assert res2 == "async-1"
    assert counter == 1

    # Test get_or_set_async with sync supplier callable
    sync_counter = 0

    def sync_creator() -> str:
        nonlocal sync_counter
        sync_counter += 1
        return f"sync-{sync_counter}"

    res3 = await cache_manager.get_or_set_async("sync_lazy", "response", sync_creator)
    res4 = await cache_manager.get_or_set_async("sync_lazy", "response", sync_creator)

    assert res3 == "sync-1"
    assert res4 == "sync-1"
    assert sync_counter == 1


def test_thread_safety_concurrency(cache_manager: CacheManager) -> None:
    """
    Spawns multiple threads to write concurrently to check that connections are
    isolated and WAL mode prevents locking issues.
    """
    errors = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(50):
                key = f"thread_key_{worker_id}_{i}"
                cache_manager.set(key, "response", f"val-{worker_id}-{i}")
                assert cache_manager.get(key) == f"val-{worker_id}-{i}"
        except Exception as e:
            errors.append(e)

    threads = []
    for t_id in range(5):
        t = threading.Thread(target=worker, args=(t_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify no exceptions occurred
    assert len(errors) == 0


def test_corrupted_json_recovery(cache_manager: CacheManager) -> None:
    # Directly write invalid JSON to the store
    cache_manager.store.set("corrupted_k", "response", "{invalid_json")

    # Get should catch JSONDecodeError, remove key, and return None
    assert cache_manager.get("corrupted_k") is None
    # Verify key was deleted
    assert cache_manager.store.get("corrupted_k") is None
