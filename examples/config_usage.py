"""
Example script demonstrating usage of the Pydantic Settings configuration module.
To run this:
    uv run python examples/config_usage.py
"""

import os


def main() -> None:
    # Inject environment overrides before the settings singleton is instantiated.
    # These must be set prior to importing `settings` because pydantic-settings
    # reads the environment exactly once at class construction time.
    os.environ["ENV"] = "staging"
    os.environ["AI__TEMPERATURE"] = "0.75"
    os.environ["DATABASE__URL"] = "sqlite:///data/example_db.db"
    os.environ["FEATURES__ENABLE_MOCK_AI"] = "True"

    # Deferred import ensures the environment is fully configured before the
    # Settings model is instantiated — avoids module-level import-order issues.
    from src.core.config import settings

    print("--- Loaded Settings ---")
    print(f"Application Name : {settings.app_name}")
    print(f"App Version      : {settings.app_version}")
    print(f"Environment      : {settings.env}")

    print("\n--- Persistence Config ---")
    print(f"SQLite DB URL    : {settings.database.url}")
    print(f"Pool Recycle     : {settings.database.pool_recycle}s")

    print("\n--- AI Engine Config ---")
    print(f"LLM Chat Model   : {settings.ai.default_chat_model}")
    print(f"LLM Temperature  : {settings.ai.temperature}")
    print(f"Ollama Base URL  : {settings.ai.ollama.base_url}")
    print(f"Ollama model     : {settings.ai.ollama.model}")

    print("\n--- Feature Toggles ---")
    print(f"Mock AI Enabled  : {settings.features.enable_mock_ai}")
    print(f"Auth Enabled     : {settings.features.enable_auth}")
    print(f"Market Sync      : {settings.features.enable_market_sync}")


if __name__ == "__main__":
    main()
