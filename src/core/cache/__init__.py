from src.core.cache.manager import CacheManager
from src.core.cache.sqlite_store import SQLiteCacheStore

# Global cache manager instance
cache_manager = CacheManager()

__all__ = [
    "CacheManager",
    "SQLiteCacheStore",
    "cache_manager",
]
