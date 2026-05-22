from pydantic import BaseModel, Field


class ScrapeGraphAIConfig(BaseModel):
    """
    Configuration for local ScrapeGraphAI structured extraction.
    """

    model: str = Field(default="llama3", description="Ollama model tag without provider prefix")
    base_url: str | None = Field(default=None, description="Ollama API base URL override")
    model_tokens: int = Field(default=4096, ge=512, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=60.0, ge=0.01, le=300.0)
    retry_attempts: int = Field(default=1, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    max_source_chars: int = Field(default=12_000, ge=1_000, le=80_000)
    verbose: bool = Field(default=False)
    headless: bool = Field(default=True)
