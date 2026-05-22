from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """
    Configuration parameters for application data persistence.
    """

    url: str = Field(
        default="sqlite:///data/career_intelligence.db", description="Database connection URL"
    )
    pool_recycle: int = Field(default=3600, ge=0)
    echo_sql: bool = Field(default=False)

    @field_validator("url")
    @classmethod
    def validate_sqlite_url(cls, v: str) -> str:
        """
        Enforce SQLite URLs for lightweight local operational design.
        """
        if not v.startswith("sqlite"):
            raise ValueError("Only SQLite database URLs are supported for this system.")
        return v
