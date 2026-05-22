from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BrowserConfig(BaseModel):
    """
    Configuration for lightweight Playwright browser automation.
    """

    browser_type: Literal["chromium", "firefox", "webkit"] = Field(default="chromium")
    headless: bool = Field(default=True)
    max_browser_instances: int = Field(default=1, ge=1, le=2)
    max_contexts: int = Field(default=2, ge=1, le=8)
    navigation_timeout_ms: int = Field(default=15_000, ge=1_000)
    action_timeout_ms: int = Field(default=10_000, ge=1_000)
    network_idle_timeout_ms: int = Field(default=5_000, ge=500)
    retry_attempts: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.25, ge=0.0, le=10.0)
    wait_until: Literal["domcontentloaded", "load", "networkidle"] = Field(
        default="domcontentloaded"
    )
    viewport_width: int = Field(default=1280, ge=320)
    viewport_height: int = Field(default=900, ge=240)
    user_agent: str | None = Field(default=None)
    java_script_enabled: bool = Field(default=True)
    block_resource_types: list[str] = Field(default_factory=list)

    @field_validator("block_resource_types")
    @classmethod
    def validate_block_resource_types(cls, values: list[str]) -> list[str]:
        allowed = {"document", "stylesheet", "image", "media", "font", "script", "xhr", "fetch"}
        normalized = [value.casefold() for value in values]
        invalid = sorted(set(normalized) - allowed)
        if invalid:
            raise ValueError(f"Unsupported browser resource types: {', '.join(invalid)}")
        return normalized
