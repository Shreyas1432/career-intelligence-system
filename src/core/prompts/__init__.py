from typing import Any

from src.core.prompts.models import PromptTemplate
from src.core.prompts.registry import (
    PromptRegistry,
    clean_name_and_extract_version,
)

# Instantiate the global prompt registry
prompt_registry = PromptRegistry()


class BackwardCompatiblePromptManager:
    """
    Bridge wrapper ensuring that applications relying on the older
    load_prompt API can operate without any code modifications.
    """

    def __init__(self, registry: PromptRegistry) -> None:
        self.registry = registry

    def load_prompt(self, relative_path: str, context: dict[str, Any] | None = None) -> str:
        """
        Backward compatible load_prompt method routing to PromptRegistry.
        """
        path_key = relative_path
        if path_key.endswith(".md"):
            path_key = path_key[:-3]

        clean_key, file_version = clean_name_and_extract_version(path_key)

        try:
            template = self.registry.get(clean_key, version=file_version)
        except KeyError as exc:
            raise FileNotFoundError(f"Prompt template file '{relative_path}' not found") from exc

        if context is None:
            return template.body

        return template.render(context)


# Global backward-compatible prompt manager instance
prompt_manager = BackwardCompatiblePromptManager(prompt_registry)

__all__ = [
    "PromptRegistry",
    "PromptTemplate",
    "prompt_manager",
    "prompt_registry",
]
