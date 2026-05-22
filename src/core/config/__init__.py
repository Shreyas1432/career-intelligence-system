from .base import PROJECT_ROOT, Settings

# Initialize the global configuration settings instance singleton
# Pydantic Settings reads from environment variables / .env automatically
settings = Settings()

__all__ = ["PROJECT_ROOT", "Settings", "settings"]
