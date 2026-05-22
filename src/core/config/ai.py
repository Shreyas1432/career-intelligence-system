from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    """
    Configuration parameters for local Ollama LLM provider.
    """

    base_url: str = Field(
        default="http://localhost:11434", description="Ollama local API server endpoint"
    )
    model: str = Field(default="llama3", description="Local model version to run")
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    model_mappings: dict[str, str] = Field(
        default_factory=lambda: {
            "fast": "llama3",
            "heavy": "mistral",
        }
    )


class ProviderConfig(BaseModel):
    """
    Configuration parameters for individual AI providers.
    """

    model: str


class AIConfig(BaseModel):
    """
    Unified parameters for Generative AI operations.
    """

    default_chat_model: str = Field(default="openai/gpt-4o-mini")
    default_embedding_model: str = Field(default="openai/text-embedding-3-small")
    max_tokens: int = Field(default=2000, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    ollama: OllamaConfig = OllamaConfig()
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @field_validator("temperature")
    @classmethod
    def validate_temp(cls, v: float) -> float:
        """
        Verify temperature bounds are within normal creativity limits.
        """
        if not (0.0 <= v <= 2.0):
            raise ValueError("Temperature must be a value between 0.0 and 2.0.")
        return v
