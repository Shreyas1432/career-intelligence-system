from typing import Any

from jinja2 import Template
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """
    Pydantic model representing a compiled versioned prompt.
    Encapsulates metadata (version, system prompt, inputs) and rendering logic.
    """

    name: str
    version: str = "1.0.0"
    description: str | None = None
    input_variables: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    body: str
    raw_content: str

    def render(self, context: dict[str, Any]) -> str:
        """
        Renders the user prompt body using the context dictionary.
        Raises ValueError if any required input variable is missing in the context.
        """
        missing = [var for var in self.input_variables if var not in context]
        if missing:
            raise ValueError(
                f"Missing required input variables for prompt '{self.name}' (v{self.version}): {missing}"
            )

        template = Template(self.body)
        return template.render(**context)

    def render_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """
        Compiles prompts into a standard chat message sequence (system prompt, user prompt).
        """
        user_content = self.render(context)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages
