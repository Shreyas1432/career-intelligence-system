import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from jinja2 import Environment, meta

from src.core.config import PROJECT_ROOT
from src.core.prompts.models import PromptTemplate

logger = structlog.get_logger("src.core.prompts.registry")

VERSION_PATTERN = re.compile(r"_v(\d+(?:\.\d+)*)$")


def clean_name_and_extract_version(file_key: str) -> tuple[str, str | None]:
    """
    Cleans prompt file keys by stripping suffixes like '_prompt' or '_template'
    and extracts trailing version annotations (e.g. 'tailor_prompt_v2' -> 'tailor', '2').
    """
    # Replace common redundant filename patterns
    normalized = file_key.replace("_prompt", "").replace("_template", "")

    # Extract version tag if it matches '_vX.Y'
    match = VERSION_PATTERN.search(normalized)
    if match:
        version = match.group(1)
        clean_key = normalized[: match.start()]
        return clean_key, version

    return normalized, None


def parse_version_tuple(v_str: str) -> tuple[int, ...]:
    """
    Parses a version string into an integer tuple for standard semantic sorting.
    """
    try:
        return tuple(int(x) for x in v_str.split("."))
    except ValueError:
        return (0,)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Splits YAML frontmatter block from raw markdown content if present.
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1])
                return metadata or {}, parts[2].strip()
            except Exception as e:
                logger.warning("Failed to parse YAML frontmatter block", error=str(e))

    return {}, content.strip()


def extract_variables(body: str) -> list[str]:
    """
    Analyses Jinja2 template AST to auto-detect undeclared placeholder variables.
    """
    env = Environment()
    try:
        ast = env.parse(body)
        return sorted(meta.find_undeclared_variables(ast))
    except Exception as e:
        logger.warning("Jinja2 template variable analysis failed", error=str(e))
        return []


class PromptRegistry:
    """
    Central repository tracking and loading versioned prompt templates.
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self.prompts_dir = prompts_dir or (PROJECT_ROOT / "prompts")
        # Structure: { name: { version: PromptTemplate } }
        self._prompts: dict[str, dict[str, PromptTemplate]] = {}
        self.load_all()

    def load_all(self) -> None:
        """
        Recursively scans prompts_dir and registers all discovered prompt markdown files.
        """
        self._prompts.clear()
        if not self.prompts_dir.exists():
            logger.warning("Prompts directory does not exist", path=str(self.prompts_dir))
            return

        for path in self.prompts_dir.rglob("*.md"):
            try:
                rel_path = path.relative_to(self.prompts_dir)
                file_key = str(rel_path.with_suffix("")).replace("\\", "/")  # Posix-compliant

                with path.open(encoding="utf-8") as f:
                    raw_content = f.read()

                self.register_prompt_from_string(file_key, raw_content)
            except Exception as e:
                logger.error("Failed to load prompt template file", path=str(path), error=str(e))

    def register_prompt_from_string(self, file_key: str, raw_content: str) -> PromptTemplate:
        """
        Parses a prompt content string, extracts metadata and versions, and registers it.
        """
        clean_name, file_version = clean_name_and_extract_version(file_key)
        metadata, body = parse_frontmatter(raw_content)

        # Priority: YAML metadata > Filename suffix > default "1.0.0"
        version = metadata.get("version") or file_version or "1.0.0"

        # Determine inputs (Priority: YAML metadata > Jinja AST auto-extraction)
        input_variables = metadata.get("input_variables")
        if input_variables is None:
            input_variables = extract_variables(body)

        system_prompt = metadata.get("system_prompt")
        description = metadata.get("description")

        template = PromptTemplate(
            name=clean_name,
            version=str(version),
            description=description,
            input_variables=input_variables,
            system_prompt=system_prompt,
            body=body,
            raw_content=raw_content,
        )

        if clean_name not in self._prompts:
            self._prompts[clean_name] = {}

        self._prompts[clean_name][str(version)] = template
        logger.debug("Registered prompt template", name=clean_name, version=version)
        return template

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """
        Retrieves a registered PromptTemplate by name and version.
        Falls back to the highest version if version is omitted or set to 'latest'.
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt template '{name}' not found in registry.")

        available = self._prompts[name]

        if not version or version == "latest":
            # Semantic sort of version keys to retrieve highest value
            sorted_keys = sorted(available.keys(), key=parse_version_tuple)
            target_version = sorted_keys[-1]
        else:
            target_version = version

        if target_version not in available:
            raise KeyError(
                f"Version '{target_version}' of prompt '{name}' is not registered. "
                f"Available versions: {list(available.keys())}"
            )

        return available[target_version]
