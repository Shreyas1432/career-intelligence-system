"""
Operational memory export synchronization layer.

Responsible for strategic, duplicate-safe exporting of curated high-significance
memories into Obsidian-compatible markdown vaults.
"""

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from src.modules.memory.repositories import MemoryRepository, MemorySummaryRepository
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemoryImportance,
    MemorySummary,
    MemoryType,
)

logger = logging.getLogger(__name__)


class VaultPathManager:
    """
    Manages Vault directories and maps MemoryEntry to deterministic paths.
    """

    def __init__(self, vault_path: Path | str) -> None:
        self.vault_path = Path(vault_path)

    def get_vault_folder(self, entry: MemoryEntry) -> Path:
        """
        Determine the folder path within the Obsidian vault for a memory entry.
        """
        content_lower = entry.content.lower()
        tags_lower = [t.lower() for t in entry.tags]

        # 11-Issues: blocker, issue, bug, problem, incident
        if (
            any(w in content_lower for w in ["issue", "bug", "problem", "incident"])
            or any(t in tags_lower for t in ["issue", "bug", "problem", "incident", "blocker"])
        ):
            folder_name = "11-Issues"
        # 04-Constraints: constraint, deadline, blocker, milestone
        elif (
            any(w in content_lower for w in ["constraint", "deadline", "blocker", "milestone"])
            or any(t in tags_lower for t in ["constraint", "deadline", "blocker", "milestone"])
        ):
            folder_name = "04-Constraints"
        # 03-Decisions: DECISION type
        elif entry.memory_type == MemoryType.DECISION:
            folder_name = "03-Decisions"
        # 00-Architecture: ARCHITECTURE domain
        elif entry.domain == MemoryDomain.ARCHITECTURE:
            folder_name = "00-Architecture"
        # 01-Relationships: RELATIONSHIP domain or recruiter keywords
        elif (
            entry.domain == MemoryDomain.RELATIONSHIP
            or any(w in content_lower for w in ["relationship", "recruiter", "hiring manager"])
            or any(t in tags_lower for t in ["relationship", "recruiter", "hiring-manager"])
        ):
            folder_name = "01-Relationships"
        # 02-Outreach: outreach, contact, cold email
        elif (
            any(w in content_lower for w in ["outreach", "cold email", "introduction"])
            or any(t in tags_lower for t in ["outreach", "cold-email", "intro"])
        ):
            folder_name = "02-Outreach"
        # 05-Lessons: lesson, learned, reflection, mistake
        else:
            folder_name = "05-Lessons"

        return self.vault_path / folder_name

    def slugify(self, text: str) -> str:
        """
        Convert text into a human-readable, safe filename slug.
        """
        first_line = text.split("\n")[0].strip()
        first_line = first_line[:40]
        slug = re.sub(r"[^\w\s-]", "", first_line).strip().lower()
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug or "memory-entry"

    def get_vault_file_path(self, entry: MemoryEntry) -> Path:
        """
        Get the deterministic file path for a MemoryEntry.
        """
        folder = self.get_vault_folder(entry)
        slug = self.slugify(entry.content)
        uuid_short = str(entry.id)[:8]
        filename = f"{slug}-{uuid_short}.md"
        return folder / filename


class ExportEligibilityEvaluator:
    """
    Evaluates whether a memory candidate is eligible for Obsidian export.
    """

    def is_eligible(self, entry: MemoryEntry) -> bool:
        """
        Determine eligibility based on curated significance requirements.
        """
        # Ignore transcripts and internal retrieval logs
        if entry.domain == MemoryDomain.RETRIEVAL:
            return False

        tags_lower = [t.lower() for t in entry.tags]
        if "transcript" in tags_lower or "temporary-log" in tags_lower:
            return False

        # Curated export only:
        # - High / Critical importance level or importance_score >= 0.7
        is_high_sig = (
            entry.importance_level in (MemoryImportance.HIGH, MemoryImportance.CRITICAL)
            or entry.importance_score >= 0.7
        )

        # - Architecture decisions/refactors
        is_arch = entry.domain == MemoryDomain.ARCHITECTURE

        # - Recruiter / Hiring manager relationship summaries
        is_rel = entry.domain == MemoryDomain.RELATIONSHIP

        # - Decisions or Summaries type
        is_dec = entry.memory_type in (MemoryType.DECISION, MemoryType.SUMMARY)

        # - Operational constraints or blocker/deadlines
        content_lower = entry.content.lower()
        is_constraint = (
            any(w in content_lower for w in ["constraint", "deadline", "blocker", "milestone"])
            or any(t in tags_lower for t in ["constraint", "deadline", "blocker", "milestone"])
        )

        # - Outreach/recruiter contact summaries
        is_outreach = (
            any(w in content_lower for w in ["outreach", "recruiter", "hiring manager", "introduction"])
            or any(t in tags_lower for t in ["outreach", "recruiter", "hiring-manager", "introduction"])
        )

        return is_high_sig or is_arch or is_rel or is_dec or is_constraint or is_outreach


class ObsidianExportFormatter:
    """
    Formats MemoryEntry and MemorySummary schemas into Obsidian-compliant markdown.
    """

    def format_entry(self, entry: MemoryEntry, summary: MemorySummary | None = None) -> str:
        """
        Format a memory entry into a markdown file string with YAML frontmatter.
        """
        first_line = entry.content.split("\n")[0].strip()
        title = first_line.lstrip("#").strip()
        if len(title) > 60:
            title = title[:57] + "..."

        frontmatter = [
            "---",
            f"id: {entry.id}",
            f"domain: {entry.domain.value}",
            f"type: {entry.memory_type.value}",
            f"source: {entry.source.value}",
            f"importance: {entry.importance_level.value}",
            f"importance_score: {entry.importance_score}",
            f"created_at: {entry.created_at.isoformat()}",
            f"updated_at: {entry.updated_at.isoformat()}",
        ]

        if entry.tags:
            frontmatter.append("tags:")
            for tag in entry.tags:
                frontmatter.append(f"  - {tag}")

        frontmatter.append("---")

        body = [
            "\n".join(frontmatter),
            f"\n# {title}\n",
            entry.content,
        ]

        if summary:
            body.append("\n## Summary")
            body.append(summary.summary_text)

            if summary.key_takeaways:
                body.append("\n## Key Takeaways")
                for takeaway in summary.key_takeaways:
                    body.append(f"- {takeaway}")

        return "\n".join(body) + "\n"


class MemoryExportService:
    """
    Coordinates duplicate-safe export of curated operational memories to an Obsidian vault.
    """

    def __init__(
        self,
        session: Session,
        vault_path: Path | str,
        evaluator: ExportEligibilityEvaluator | None = None,
        formatter: ObsidianExportFormatter | None = None,
        path_manager: VaultPathManager | None = None,
    ) -> None:
        self.session = session
        self.vault_path = Path(vault_path)
        self.evaluator = evaluator or ExportEligibilityEvaluator()
        self.formatter = formatter or ObsidianExportFormatter()
        self.path_manager = path_manager or VaultPathManager(vault_path)

        self.memory_repo = MemoryRepository(session)
        self.summary_repo = MemorySummaryRepository(session)

    def export_all(self) -> int:
        """
        Export all eligible memories to the vault.
        Returns the count of successfully synchronized/written files.
        """
        entries = self.memory_repo.list_entries(limit=10000)
        return self._export_batch(entries)

    def export_changes_since(self, since: datetime) -> int:
        """
        Export memories created or updated since the specified timestamp.
        """
        all_entries = self.memory_repo.list_entries(limit=10000)

        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)

        filtered = []
        for entry in all_entries:
            updated_at = entry.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            if updated_at >= since:
                filtered.append(entry)

        return self._export_batch(filtered)

    def export_entry_by_id(self, entry_id: UUID) -> bool:
        """
        Export a single memory entry by its UUID.
        Returns True if exported successfully, False if not found or not eligible.
        """
        entry = self.memory_repo.get_by_uuid(entry_id)
        if not entry:
            logger.warning(f"Memory entry {entry_id} not found in database.")
            return False

        if not self.evaluator.is_eligible(entry):
            logger.info(f"Memory entry {entry_id} is not eligible for export.")
            return False

        return self._write_entry(entry)

    def _export_batch(self, entries: Sequence[MemoryEntry]) -> int:
        """
        Process and write a batch of memory entries.
        """
        success_count = 0
        for entry in entries:
            if not self.evaluator.is_eligible(entry):
                continue

            try:
                if self._write_entry(entry):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to export entry {entry.id}: {e}", exc_info=True)

        return success_count

    def _write_entry(self, entry: MemoryEntry) -> bool:
        """
        Write formatted markdown to file deterministically and duplicate-safely.
        Returns True if written/updated, False if skipped because no change.
        """
        file_path = self.path_manager.get_vault_file_path(entry)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.summary_repo.get_by_memory_id(entry.id)
        formatted_content = self.formatter.format_entry(entry, summary)

        if file_path.exists():
            existing_content = file_path.read_text(encoding="utf-8")
            if existing_content == formatted_content:
                logger.debug(f"Skipped duplicate write for {file_path.name}")
                return False

        file_path.write_text(formatted_content, encoding="utf-8")
        logger.info(f"Synchronized memory {entry.id} to vault path: {file_path}")
        return True
