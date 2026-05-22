from pydantic import BaseModel, Field


class FeatureFlags(BaseModel):
    """
    Feature toggle switches to enable/disable app flows easily.
    """

    enable_mock_ai: bool = Field(
        default=False,
        description="Bypasses call endpoints, serving static strings for local UI tests",
    )
    enable_auth: bool = Field(default=False, description="Toggle registration/login screens")
    enable_market_sync: bool = Field(
        default=True, description="Allows scraping/polling job search platforms"
    )
