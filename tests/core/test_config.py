import pytest
from pydantic import ValidationError

from src.core.config.base import Settings


def test_default_config_values():
    """
    Ensure the configuration module loads safe, default parameters when unconfigured.
    """
    config = Settings()

    assert config.env == "development"
    assert config.database.pool_recycle == 3600
    assert "sqlite" in config.database.url
    assert config.ai.temperature == 0.2
    assert config.browser.headless is True
    assert config.browser.max_browser_instances <= 2
    assert config.scrapegraphai.model == "llama3"
    assert config.scrapegraphai.retry_attempts == 1
    assert config.features.enable_mock_ai is False


def test_database_url_validation():
    """
    Validate that database URLs are strictly checked for SQLite support.
    """
    # SQLite URL should pass
    valid_settings = Settings(database={"url": "sqlite:///data/test.db"})
    assert valid_settings.database.url == "sqlite:///data/test.db"

    # PostgreSQL or other URL schemas should fail validation
    with pytest.raises(ValidationError) as exc_info:
        Settings(database={"url": "postgresql://user:pass@host:5432/db"})

    assert "sqlite" in str(exc_info.value).lower()


def test_ai_temperature_validation():
    """
    Verify boundary checks on AI model temperature arguments.
    """
    # Verify values inside range [0.0, 2.0]
    config = Settings(ai={"temperature": 1.5})
    assert config.ai.temperature == 1.5

    # Check value under range limit
    with pytest.raises(ValidationError) as exc_info:
        Settings(ai={"temperature": -0.1})
    assert "temperature" in str(exc_info.value)

    # Check value over range limit
    with pytest.raises(ValidationError) as exc_info:
        Settings(ai={"temperature": 2.1})
    assert "temperature" in str(exc_info.value)


def test_environment_variable_override(monkeypatch):
    """
    Confirm environment overrides load correctly via double-underscore nested fields.
    """
    monkeypatch.setenv("DATABASE__URL", "sqlite:///data/overridden.db")
    monkeypatch.setenv("AI__TEMPERATURE", "0.89")
    monkeypatch.setenv("BROWSER__HEADLESS", "False")
    monkeypatch.setenv("SCRAPEGRAPHAI__MODEL", "mistral")
    monkeypatch.setenv("FEATURES__ENABLE_MOCK_AI", "True")
    monkeypatch.setenv("ENV", "production")

    config = Settings()

    assert config.database.url == "sqlite:///data/overridden.db"
    assert config.ai.temperature == 0.89
    assert config.browser.headless is False
    assert config.scrapegraphai.model == "mistral"
    assert config.features.enable_mock_ai is True
    assert config.env == "production"
