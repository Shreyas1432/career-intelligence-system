"""
Operational memory ingestion layer.

Provides preprocessors, metadata classifiers, compression coordinators,
and ingestion services for curated markdown notes.
"""

import re
from enum import StrEnum
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.modules.memory.persistence import (
    MemoryPersistenceService,
    PersistenceEligibility,
    PersistenceResult,
)
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemorySource,
    MemorySummary,
    MemoryType,
)


class IngestionStatus(StrEnum):
    """Status outcomes for markdown note ingestion requests."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class PreprocessedNote(BaseModel):
    """Intermediate schema containing parsed and normalized candidate details."""

    clean_content: str = Field(..., description="Normalized body content.")
    raw_metadata: dict[str, Any] = Field(default_factory=dict, description="Raw YAML frontmatter.")
    domain: MemoryDomain = Field(..., description="Resolved domain classification.")
    memory_type: MemoryType = Field(..., description="Resolved structural memory type.")
    source: MemorySource = Field(..., description="Resolved origin source.")
    tags: list[str] = Field(default_factory=list, description="Extracted categorization tags.")
    extracted_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Remaining metadata key-value parameters."
    )


class IngestionResult(BaseModel):
    """Result response for a markdown ingestion request."""

    status: IngestionStatus = Field(..., description="Outcome of the ingestion.")
    entry: MemoryEntry | None = Field(
        default=None, description="Persisted entry schema on success."
    )
    summary: MemorySummary | None = Field(
        default=None, description="Persisted compressed summary on success."
    )
    persistence_result: PersistenceResult | None = Field(
        default=None, description="Full persistence outcome details."
    )
    explanation: str = Field(..., description="Explanation of the ingestion outcome.")


class IngestionSourceClassifier:
    """
    Classifies memory domain, type, and source based on content and file context.
    """

    def classify(
        self,
        content: str,
        file_path: str | None = None,
        file_name: str | None = None,
    ) -> tuple[MemoryDomain, MemoryType, MemorySource]:
        """
        Classifies the domain, type, and source based on path, name, or content indicators.
        """
        path_lower = (file_path or "").lower()
        name_lower = (file_name or "").lower()
        content_lower = content.lower()
        combined = f"{path_lower}/{name_lower}"

        source = self._classify_source(path_lower, name_lower)
        domain = self._classify_domain(combined, content_lower)
        memory_type = self._classify_type(combined, content_lower)

        return domain, memory_type, source

    @staticmethod
    def _classify_source(path_lower: str, name_lower: str) -> MemorySource:
        if "obsidian" in path_lower or "obsidian" in name_lower:
            return MemorySource.OBSIDIAN
        if "user" in path_lower or "user" in name_lower:
            return MemorySource.USER
        return MemorySource.INGESTION

    @staticmethod
    def _classify_domain(combined: str, content_lower: str) -> MemoryDomain:
        # Check path first
        if any(w in combined for w in ["architecture", "refactor", "design"]):
            return MemoryDomain.ARCHITECTURE
        if any(
            w in combined
            for w in ["relationship", "recruiter", "contact", "outreach", "hiring", "interviewer"]
        ):
            return MemoryDomain.RELATIONSHIP
        if any(w in combined for w in ["retrieval", "search", "embedding", "rerank"]):
            return MemoryDomain.RETRIEVAL
        if any(w in combined for w in ["codebase", "monorepo", "repository", "module"]):
            return MemoryDomain.CODEBASE
        if any(w in combined for w in ["constraint", "deadline", "operational", "sprint"]):
            return MemoryDomain.OPERATIONAL

        # Check content fallback
        if any(
            w in content_lower
            for w in ["architecture", "design decision", "monorepo", "migration"]
        ):
            return MemoryDomain.ARCHITECTURE
        if any(w in content_lower for w in ["recruiter", "hiring manager", "outreach", "linkedin"]):
            return MemoryDomain.RELATIONSHIP
        if any(w in content_lower for w in ["constraint", "deadline", "milestone", "sprint"]):
            return MemoryDomain.OPERATIONAL
        if any(w in content_lower for w in ["retrieval", "embedding", "vector", "cosine"]):
            return MemoryDomain.RETRIEVAL

        return MemoryDomain.OPERATIONAL

    @staticmethod
    def _classify_type(combined: str, content_lower: str) -> MemoryType:
        if "summary" in combined:
            return MemoryType.SUMMARY
        if "decision" in combined:
            return MemoryType.DECISION
        if "metadata" in combined:
            return MemoryType.METADATA
        if "fact" in combined:
            return MemoryType.FACT

        if any(w in content_lower for w in ["decision", "decided", "conclude"]):
            return MemoryType.DECISION
        if any(w in content_lower for w in ["summary", "summarize", "overview"]):
            return MemoryType.SUMMARY

        return MemoryType.FACT


class MemoryPreprocessor:
    """
    Normalizes markdown text and extracts frontmatter metadata tags.
    """

    def __init__(self) -> None:
        self.classifier = IngestionSourceClassifier()

    def preprocess(
        self,
        raw_markdown: str,
        *,
        file_path: str | None = None,
        file_name: str | None = None,
    ) -> PreprocessedNote:
        """
        Normalizes body text, parses YAML frontmatter, and resolves enums.
        """
        metadata, body = self._parse_frontmatter(raw_markdown)

        # Get defaults/resolved fields from classifier for whatever is missing
        c_domain, c_type, c_source = self.classifier.classify(
            body, file_path=file_path, file_name=file_name
        )

        resolved_domain, resolved_type, resolved_source = self._resolve_enums(
            metadata, c_domain, c_type, c_source
        )
        tags = self._extract_tags(metadata)

        # Extract remaining metadata dict
        extracted_metadata = {
            k: v
            for k, v in metadata.items()
            if k not in ["domain", "type", "memory_type", "source", "tags"]
        }

        return PreprocessedNote(
            clean_content=body,
            raw_metadata=metadata,
            domain=resolved_domain,
            memory_type=resolved_type,
            source=resolved_source,
            tags=tags,
            extracted_metadata=extracted_metadata,
        )

    @staticmethod
    def _parse_frontmatter(raw_markdown: str) -> tuple[dict[str, Any], str]:
        content = raw_markdown.strip()
        metadata: dict[str, Any] = {}
        body = content

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if match:
            yaml_content = match.group(1)
            body = content[match.end() :].strip()
            try:
                parsed = yaml.safe_load(yaml_content)
                if isinstance(parsed, dict):
                    metadata = parsed
            except yaml.YAMLError:
                pass

        return metadata, body

    @staticmethod
    def _resolve_enums(
        metadata: dict[str, Any],
        c_domain: MemoryDomain,
        c_type: MemoryType,
        c_source: MemorySource,
    ) -> tuple[MemoryDomain, MemoryType, MemorySource]:
        resolved_domain = c_domain
        if "domain" in metadata:
            try:
                resolved_domain = MemoryDomain(str(metadata["domain"]).lower().strip())
            except ValueError:
                pass

        resolved_type = c_type
        if "type" in metadata or "memory_type" in metadata:
            type_val = metadata.get("type") or metadata.get("memory_type")
            try:
                resolved_type = MemoryType(str(type_val).lower().strip())
            except ValueError:
                pass

        resolved_source = c_source
        if "source" in metadata:
            try:
                resolved_source = MemorySource(str(metadata["source"]).lower().strip())
            except ValueError:
                pass

        return resolved_domain, resolved_type, resolved_source

    @staticmethod
    def _extract_tags(metadata: dict[str, Any]) -> list[str]:
        tags = []
        if "tags" in metadata:
            tags_val = metadata["tags"]
            if isinstance(tags_val, list):
                tags = [str(t).strip() for t in tags_val]
            elif isinstance(tags_val, str):
                tags = [t.strip() for t in tags_val.split(",") if t.strip()]
        return tags


class MemoryCompressionCoordinator:
    """
    Coordinates heuristic generation of summaries and key takeaways from memory content.
    """

    def prepare_candidate_summary(
        self, content: str, memory_id: UUID | None = None
    ) -> MemorySummary:
        """
        Extracts takeaways and generates a summary text from raw content.
        """
        lines = [line.strip() for line in content.split("\n")]
        takeaways = []
        summary_lines = []

        for line in lines:
            if line.startswith(("-", "*", "1.", "2.", "3.")) and len(line) > 2:
                # Strip list prefix
                clean_line = re.sub(r"^[-*\d\.\s]+", "", line).strip()
                if clean_line:
                    takeaways.append(clean_line)
            elif line.startswith("#") or not line:
                continue
            else:
                summary_lines.append(line)

        if not takeaways:
            # Fallback: look for action/constraint indicators
            for line in lines:
                if any(kw in line for kw in ["Action:", "Decision:", "Constraint:", "Note:"]):
                    takeaways.append(line)

        # Extract summary text from non-list lines
        base_summary = " ".join(summary_lines[:5]).strip()
        if not base_summary:
            base_summary = content[:1500].strip()

        summary_text = base_summary[:2000]

        return MemorySummary(
            memory_id=memory_id,
            summary_text=summary_text,
            original_length=len(content),
            compressed_length=len(summary_text),
            key_takeaways=takeaways[:5],
        )


class MemoryIngestionService:
    """
    Service coordinating markdown memory ingestion and classification.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.persistence_service = MemoryPersistenceService(session)
        self.preprocessor = MemoryPreprocessor()
        self.compressor = MemoryCompressionCoordinator()

    def ingest_markdown(
        self,
        raw_markdown: str,
        *,
        file_path: str | None = None,
        file_name: str | None = None,
    ) -> IngestionResult:
        """
        Preprocesses, classifies, filters, and persists markdown memory content.
        """
        if not raw_markdown or not raw_markdown.strip():
            return IngestionResult(
                status=IngestionStatus.FAILED,
                explanation="Ingestion failed: empty content provided.",
            )

        # 1. Preprocess note (normalization, YAML extraction, source classification)
        preprocessed = self.preprocessor.preprocess(
            raw_markdown, file_path=file_path, file_name=file_name
        )

        # 2. Check for transcripts indicator or giant blob constraints early
        content_lower = preprocessed.clean_content.lower()
        transcript_signals = ["transcript:", "speaker:", "interviewer:", "session log"]
        if any(sig in content_lower for sig in transcript_signals):
            return IngestionResult(
                status=IngestionStatus.SKIPPED,
                explanation="Ingestion skipped: transcripts are not allowed.",
            )

        if len(preprocessed.clean_content) > 8000:
            return IngestionResult(
                status=IngestionStatus.SKIPPED,
                explanation="Ingestion skipped: content length exceeds 8000 character blob limit.",
            )

        # 3. Invoke persistence layer
        persist_res = self.persistence_service.persist(
            content=preprocessed.clean_content,
            domain=preprocessed.domain,
            memory_type=preprocessed.memory_type,
            source=preprocessed.source,
            tags=preprocessed.tags,
            metadata=preprocessed.extracted_metadata,
        )

        if (
            persist_res.eligibility == PersistenceEligibility.ELIGIBLE
            and persist_res.entry is not None
        ):
            # 4. Generate compressed candidate summary
            summary = self.compressor.prepare_candidate_summary(
                preprocessed.clean_content, memory_id=persist_res.entry.id
            )
            # 5. Persist the summary
            saved_summary = self.persistence_service.summary_repository.save_summary(summary)

            return IngestionResult(
                status=IngestionStatus.SUCCESS,
                entry=persist_res.entry,
                summary=saved_summary,
                persistence_result=persist_res,
                explanation="Ingested successfully.",
            )

        elif persist_res.eligibility == PersistenceEligibility.REVIEW_REQUIRED:
            return IngestionResult(
                status=IngestionStatus.SKIPPED,
                persistence_result=persist_res,
                explanation=f"Ingestion skipped (needs review): {persist_res.explanation}",
            )

        # PersistenceEligibility.INELIGIBLE
        return IngestionResult(
            status=IngestionStatus.SKIPPED,
            persistence_result=persist_res,
            explanation=f"Ingestion rejected: {persist_res.explanation}",
        )


__all__ = [
    "IngestionResult",
    "IngestionSourceClassifier",
    "IngestionStatus",
    "MemoryCompressionCoordinator",
    "MemoryIngestionService",
    "MemoryPreprocessor",
    "PreprocessedNote",
]
