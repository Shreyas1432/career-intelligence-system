"""
Example script demonstrating usage of the Pydantic Settings configuration module.
To run this:
    uv run python examples/config_usage.py
"""
import os
import sys
from pathlib import Path

# Setup system paths for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Simulating environment variables before config load
os.environ["ENV"] = "staging"
os.environ["AI__TEMPERATURE"] = "0.75"
os.environ["DATABASE__URL"] = "sqlite:///data/example_db.db"
os.environ["FEATURES__ENABLE_MOCK_AI"] = "True"

from src.core.config import settings


def main():
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
