from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from .ai import AIConfig
from .browser import BrowserConfig
from .cache import CacheConfig
from .db import DatabaseConfig
from .features import FeatureFlags
from .logging_cfg import LoggingConfig
from .scrapegraphai import ScrapeGraphAIConfig

# Locate project base directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class AppConfig(BaseModel):
    name: str = "Career Intelligence System"
    version: str = "0.1.0"
    env: str = "development"


class ResumeModuleConfig(BaseModel):
    max_upload_size_mb: int = 5
    allowed_extensions: list[str] = [".pdf", ".docx", ".txt"]


class InterviewModuleConfig(BaseModel):
    max_history_sessions: int = 10


class ModulesConfig(BaseModel):
    resume: ResumeModuleConfig = ResumeModuleConfig()
    interview: InterviewModuleConfig = InterviewModuleConfig()


class Settings(BaseSettings):
    """
    Main Settings class compiling all modular nested configs.
    Supports env file resolution and system environment overrides using double-underscore syntax.
    Example environment variable overrides:
      - DATABASE__URL=sqlite:///data/prod.db (Sets database.url)
      - AI__TEMPERATURE=0.8 (Sets ai.temperature)
      - FEATURES__ENABLE_MOCK_AI=true (Sets features.enable_mock_ai)
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file=str(PROJECT_ROOT / "config" / "settings.yaml"),
        extra="ignore",
    )

    env: str = "development"
    app_name: str = "Career Intelligence System"
    app_version: str = "0.1.0"
    secret_key: str = "secure-default-change-in-production"

    # Cloud LLM provider keys (loaded directly from environment variables)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_api_base: str | None = None

    # Nested configurations
    app: AppConfig = AppConfig()
    database: DatabaseConfig = DatabaseConfig()
    ai: AIConfig = AIConfig()
    browser: BrowserConfig = BrowserConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    features: FeatureFlags = FeatureFlags()
    scrapegraphai: ScrapeGraphAIConfig = ScrapeGraphAIConfig()
    modules: ModulesConfig = ModulesConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
