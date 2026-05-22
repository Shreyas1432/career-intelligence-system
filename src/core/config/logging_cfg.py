from typing import Literal

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """
    Configuration parameters for logger handlers.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        description="Standard formatting pattern",
    )
    file_path: str = Field(default="data/app.log")
