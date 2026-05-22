import sqlite3
import threading
import time
from pathlib import Path
from typing import cast

import structlog

from src.core.config import PROJECT_ROOT, settings

logger = structlog.get_logger("src.core.cache.sqlite_store")


class SQLiteCacheStore:
    """
    Low-overhead, thread-safe SQLite key-value store using direct sqlite3 connection.
    Avoids ORM dependencies to optimize memory footprint and execution speed.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = settings.cache.db_path

        path = Path(db_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        self.db_path = path
        self._local = threading.local()

        # Ensure parent folder exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema and index configuration
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """
        Retrieves a thread-isolated SQLite connection.
        Enables performance pragmas for MacBook Air.
        """
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    def _init_db(self) -> None:
        """
        Initializes schema tables and indexes.
        """
        conn = self._get_conn()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value_type TEXT NOT NULL,
                        value TEXT NOT NULL,
                        expires_at REAL,
                        created_at REAL NOT NULL
                    )
                    """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_value_type ON cache_entries(value_type)"
                )
        except Exception as e:
            logger.error("Failed to initialize SQLite cache tables", error=str(e))
            raise

        # Perform initial cleanup of expired entries
        self.prune_expired()

    def prune_expired(self) -> None:
        """
        Removes all expired entries from cache.
        """
        conn = self._get_conn()
        now = time.time()
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.debug("Pruned expired cache entries", count=deleted)
        except Exception as e:
            logger.warning("Failed to prune expired cache entries", error=str(e))

    def get(self, key: str) -> str | None:
        """
        Retrieves a value by key. Handles inline TTL expiration.
        """
        conn = self._get_conn()
        now = time.time()
        try:
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row is None:
                return None

            val, expires_at = row
            if expires_at is not None and now > expires_at:
                # Delete inline if expired
                self.delete(key)
                return None

            return cast(str, val)
        except Exception as e:
            logger.error("Failed to read from SQLite cache", key=key, error=str(e))
            return None

    def set(self, key: str, value_type: str, value: str, ttl_seconds: int | None = None) -> None:
        """
        Inserts or replaces a cache key-value pair.
        """
        conn = self._get_conn()
        now = time.time()
        expires_at = (now + ttl_seconds) if ttl_seconds is not None else None

        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries (key, value_type, value, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, value_type, value, expires_at, now),
                )
        except Exception as e:
            logger.error("Failed to write to SQLite cache", key=key, error=str(e))

    def delete(self, key: str) -> bool:
        """
        Deletes a cache key-value pair. Returns True if found and deleted.
        """
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Failed to delete from SQLite cache", key=key, error=str(e))
            return False

    def delete_by_prefix(self, prefix: str) -> int:
        """
        Deletes any keys starting with the specified prefix. Returns count of deleted rows.
        """
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key LIKE ?", (f"{prefix}%",))
                return cursor.rowcount
        except Exception as e:
            logger.error(
                "Failed to delete by prefix from SQLite cache",
                prefix=prefix,
                error=str(e),
            )
            return 0

    def clear(self) -> None:
        """
        Clears all items from the cache database.
        """
        conn = self._get_conn()
        try:
            with conn:
                conn.execute("DELETE FROM cache_entries")
        except Exception as e:
            logger.error("Failed to clear SQLite cache", error=str(e))

    def close(self) -> None:
        """
        Closes the thread-local connection.
        """
        if hasattr(self._local, "conn"):
            try:
                self._local.conn.close()
            except Exception:
                pass
            del self._local.conn
