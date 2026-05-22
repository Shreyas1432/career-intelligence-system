import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from src.core.cache.sqlite_store import SQLiteCacheStore
from src.core.config import settings

logger = structlog.get_logger("src.core.cache.manager")


class CacheManager:
    """
    Orchestration layer managing cache key generation, serialization,
    and thread-safe synchronous and asynchronous caching operations.
    """

    def __init__(self, store: SQLiteCacheStore | None = None) -> None:
        self.store = store or SQLiteCacheStore()

    def generate_embedding_key(self, model: str, text: str) -> str:
        """
        Generates a unique hashed cache key for an embedding request.
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{model}:{text_hash}"

    def generate_ai_response_key(
        self,
        model: str,
        system_prompt: str | None,
        prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a unique hashed cache key for a completion response.
        Ensures consistent serialization of payload parameters using sorted keys.
        """
        payload = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        }
        serialized = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"ai_response:{model}:{payload_hash}"

    def get(self, key: str) -> Any | None:
        """
        Retrieves and deserializes cached content from SQLite.
        Returns None on miss or if caching is disabled.
        """
        if not settings.cache.enabled:
            return None

        raw = self.store.get(key)
        if raw is None:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to deserialize cache value, removing corrupt key",
                key=key,
                error=str(exc),
            )
            self.delete(key)
            return None

    def set(self, key: str, value_type: str, value: Any, ttl_seconds: int | None = None) -> None:
        """
        Serializes and persists value in the cache store.
        """
        if not settings.cache.enabled:
            return

        try:
            serialized = json.dumps(value)
            self.store.set(key, value_type, serialized, ttl_seconds)
        except Exception as exc:
            logger.error("Failed to serialize cache value", key=key, error=str(exc))

    def get_or_set(
        self,
        key: str,
        value_type: str,
        creator_fn: Callable[[], Any],
        ttl_seconds: int | None = None,
    ) -> Any:
        """
        Retrieves cached entry. Evaluates and stores creator_fn on a cache miss.
        """
        if not settings.cache.enabled:
            return creator_fn()

        cached = self.get(key)
        if cached is not None:
            return cached

        val = creator_fn()
        self.set(key, value_type, val, ttl_seconds)
        return val

    def delete(self, key: str) -> bool:
        """
        Explicitly removes a cache key.
        """
        return self.store.delete(key)

    def delete_by_prefix(self, prefix: str) -> int:
        """
        Invalidates cached keys sharing a namespace prefix.
        """
        return self.store.delete_by_prefix(prefix)

    def clear(self) -> None:
        """
        Deletes all cache keys.
        """
        self.store.clear()

    # --- Async APIs ---

    async def get_async(self, key: str) -> Any | None:
        """
        Asynchronously retrieves and deserializes cached content.
        Uses thread-pool delegation to avoid blocking the event loop.
        """
        if not settings.cache.enabled:
            return None
        return await asyncio.to_thread(self.get, key)

    async def set_async(
        self, key: str, value_type: str, value: Any, ttl_seconds: int | None = None
    ) -> None:
        """
        Asynchronously serializes and writes a cached value.
        """
        if not settings.cache.enabled:
            return
        await asyncio.to_thread(self.set, key, value_type, value, ttl_seconds)

    async def get_or_set_async(
        self,
        key: str,
        value_type: str,
        creator_fn: Callable[[], Awaitable[Any] | Any],
        ttl_seconds: int | None = None,
    ) -> Any:
        """
        Asynchronously retrieves or populates a cache entry.
        Supports both sync and async callable suppliers.
        """
        if not settings.cache.enabled:
            res = creator_fn()
            if inspect.isawaitable(res):
                return await res
            return res

        cached = await self.get_async(key)
        if cached is not None:
            return cached

        res = creator_fn()
        if inspect.isawaitable(res):
            val = await res
        else:
            val = res

        await self.set_async(key, value_type, val, ttl_seconds)
        return val
