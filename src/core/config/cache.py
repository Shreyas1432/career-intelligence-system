from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """
    Configuration parameters for the lightweight local SQLite caching layer.
    """

    enabled: bool = Field(default=True, description="Enables or disables the local cache layer")
    db_path: str = Field(
        default="data/cache.db", description="Path to the SQLite cache database file"
    )
    default_ttl_seconds: int = Field(
        default=86400, ge=0, description="Default TTL for AI responses (24 hours)"
    )
    embedding_ttl_seconds: int = Field(
        default=604800, ge=0, description="Default TTL for embeddings (7 days)"
    )
