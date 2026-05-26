"""
Lightweight performance benchmarking script for the career intelligence platform.

Measures latency across key workflows:
1. Relationship creation
2. Outreach tracking
3. Follow-up recommendation generation
4. Memory ingestion
5. Significance evaluation
6. Embedding generation
7. Semantic retrieval
8. Retrieval reranking
9. Token-efficient context assembly
10. Export synchronization

Target performance thresholds:
- Ingestion: <300ms
- Retrieval: <500ms
- Reranking: <200ms
- Context assembly: <200ms
- Follow-up recommendation: <500ms

Strict design constraints:
- Local-first in-memory SQLite database.
- Deterministic 384-dimensional mock embedding provider.
- No external network or live LLM service calls.
- Clear reporting of average/max latency and pass/fail thresholds.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

# Adjust Python path to allow execution from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.database.models import Base
from src.modules.memory.embeddings import EmbeddingProvider, EmbeddingService
from src.modules.memory.export_sync import MemoryExportService
from src.modules.memory.ingestion import MemoryIngestionService
from src.modules.memory.ranking import RetrievalRankingService
from src.modules.memory.retrieval import ContextAssembler, MemoryRetrievalService
from src.modules.memory.schemas import (
    MemoryDomain,
    MemoryEntry,
    MemoryRetrievalResult,
    MemorySource,
    MemoryType,
)
from src.modules.memory.significance import SignificanceEvaluator
from src.modules.relationship import (
    ContactCreate,
    ContactService,
    ContactType,
    FollowupRecommendationService,
    OutreachTrackingService,
)


class _MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic 384-dimensional mock embedding provider for benchmarking.

    Computes normalized vectors deterministically derived from text length to
    model the performance overhead of storage/retrieval without external model loading.
    """

    _DIMENSION: int = 384

    @property
    def model_name(self) -> str:
        return "mock-384d-bench"

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            # Generate deterministic values based on text length to avoid overhead
            val = (len(text) % 100 + 1) / 100.0
            # Build a normalized vector of length self._dimension
            vec = [val * (i + 1) for i in range(self._DIMENSION)]
            # Normalize vector (L2 norm)
            norm = sum(x * x for x in vec) ** 0.5
            normalized = [x / norm for x in vec] if norm > 0.0 else [0.0] * self._DIMENSION
            results.append(normalized)
        return results


def benchmark_workflow(
    name: str,
    fn: Callable[[], Any],
    iterations: int = 50,
    warmup: int = 5,
) -> dict[str, Any]:
    """Helper to run a workflow function multiple times and compile latency stats."""
    # Warmup runs to stabilize JVM/interpreter and db session state
    for _ in range(warmup):
        fn()

    # Latency tracking
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    return {
        "name": name,
        "avg": statistics.mean(latencies),
        "max": max(latencies),
        "min": min(latencies),
    }


# ---------------------------------------------------------------------------
# Workflow runners builders
# ---------------------------------------------------------------------------


def make_relationship_creation_runner(contact_service: ContactService) -> Callable[[], None]:
    """Runner for benchmarking contact creation."""
    index = 0

    def run() -> None:
        nonlocal index
        index += 1
        contact_service.create_contact(
            ContactCreate(
                first_name=f"BenchJordan_{index}",
                last_name="Lee",
                company="Stripe",
                contact_type=ContactType.RECRUITER,
                email=f"jordan_bench_{index}@stripe.com",
            )
        )

    return run


def make_outreach_tracking_runner(
    tracking_service: OutreachTrackingService, contact_id: UUID
) -> Callable[[], None]:
    """Runner for benchmarking outreach logging and state sync."""
    index = 0

    def run() -> None:
        nonlocal index
        index += 1
        tracking_service.log_outreach_event(
            contact_id=contact_id,
            method="LinkedIn",
            content=f"Performance check message outreach #{index}.",
        )
        tracking_service._sync_relationship_state(contact_id)

    return run


def make_followup_rec_runner(
    followup_service: FollowupRecommendationService,
) -> Callable[[], None]:
    """Runner for benchmarking follow-up recommendation generation."""

    def run() -> None:
        followup_service.generate_recommendations()

    return run


def make_memory_ingestion_runner(ingestion_service: MemoryIngestionService) -> Callable[[], None]:
    """Runner for benchmarking markdown memory ingestion."""
    note_content = """---
domain: relationship
type: fact
source: user
tags: [recruiter, stripe, performance]
---
# Recruiter Jordan Lee
Jordan Lee is a technical recruiter at Stripe. Prefers Slack follow-up.
Warm referral from the hiring manager is available.
"""

    def run() -> None:
        ingestion_service.ingest_markdown(note_content)

    return run


def make_significance_evaluation_runner(evaluator: SignificanceEvaluator) -> Callable[[], None]:
    """Runner for benchmarking significance evaluation."""
    content = (
        "SQLite WAL mode architecture decision: adopt WAL mode for the bounded "
        "context database to enable concurrent read/write without contention."
    )

    def run() -> None:
        evaluator.evaluate(
            content=content,
            domain=MemoryDomain.ARCHITECTURE,
            memory_type=MemoryType.DECISION,
            source=MemorySource.USER,
        )

    return run


def make_embedding_generation_runner(
    emb_service: EmbeddingService, entry: MemoryEntry
) -> Callable[[], None]:
    """Runner for benchmarking embedding generation."""

    def run() -> None:
        emb_service.generate_embedding(entry)

    return run


def make_semantic_retrieval_runner(retrieval_service: MemoryRetrievalService) -> Callable[[], None]:
    """Runner for benchmarking semantic context retrieval."""

    def run() -> None:
        retrieval_service.retrieve_context(
            query="sqlite performance concurrency",
            domain=MemoryDomain.ARCHITECTURE,
            min_similarity=0.0,
            limit=5,
        )

    return run


def make_retrieval_reranking_runner(
    ranking_service: RetrievalRankingService,
    results: list[MemoryRetrievalResult],
) -> Callable[[], None]:
    """Runner for benchmarking retrieval candidates reranking."""

    def run() -> None:
        ranking_service.rerank(results, limit=5)

    return run


def make_context_assembly_runner(
    assembler: ContextAssembler,
    results: list[MemoryRetrievalResult],
) -> Callable[[], None]:
    """Runner for benchmarking token-efficient context assembly."""

    def run() -> None:
        assembler.assemble(
            query="sqlite performance concurrency",
            results=results,
            max_chars=4000,
            domain_filters=[MemoryDomain.ARCHITECTURE],
        )

    return run


def make_export_sync_runner(export_service: MemoryExportService) -> Callable[[], None]:
    """Runner for benchmarking Obsidian vault export synchronization."""

    def run() -> None:
        export_service.export_all()

    return run


# ---------------------------------------------------------------------------
# Seeding and orchestration
# ---------------------------------------------------------------------------


def seed_data(
    session: Session,
    contact_service: ContactService,
    tracking_service: OutreachTrackingService,
    ingestion_service: MemoryIngestionService,
    emb_service: EmbeddingService,
) -> tuple[UUID, MemoryEntry, list[MemoryRetrievalResult]]:
    """Seeds the SQLite database to prepare for realistic benchmark workflows."""
    # 1. Create a contact and log initial interaction
    contact = contact_service.create_contact(
        ContactCreate(
            first_name="SeedJordan",
            last_name="Lee",
            company="Stripe",
            contact_type=ContactType.RECRUITER,
            email="jordan_seed@stripe.com",
        )
    )
    tracking_service.log_outreach_event(
        contact_id=contact.id,
        method="LinkedIn",
        content="Hello Jordan, nice to connect.",
    )
    tracking_service._sync_relationship_state(contact.id)

    # 2. Ingest several memories to query during retrieval benchmarks
    memories = []
    for i in range(10):
        note = f"""---
domain: architecture
type: decision
source: user
tags: [sqlite, perf, seed]
---
# SQLite WAL Mode Decision {i}
This is seed memory #{i} detailing bounded domain architecture decision for sqlite.
"""
        res = ingestion_service.ingest_markdown(note)
        if res.entry:
            memories.append(res.entry)

    # 3. Generate embeddings for the memories
    emb_service.orchestrate_embeddings(session, memories)

    # Return key seeded objects
    sample_entry = memories[0]

    # Get retrieval candidates
    retrieval_service = MemoryRetrievalService(session, emb_service)
    ctx = retrieval_service.retrieve_context(
        "sqlite performance concurrency",
        domain=MemoryDomain.ARCHITECTURE,
        min_similarity=0.0,
        limit=10,
    )

    return contact.id, sample_entry, ctx.results


def run_benchmarks() -> None:
    """Orchestrates benchmarking of the 10 career intelligence platform workflows."""
    print("Initializing in-memory SQLite database...")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    session = session_local()

    # Initialize services
    contact_service = ContactService(session)
    tracking_service = OutreachTrackingService(session)
    followup_service = FollowupRecommendationService(session)
    ingestion_service = MemoryIngestionService(session)
    evaluator = SignificanceEvaluator()

    mock_provider = _MockEmbeddingProvider()
    emb_service = EmbeddingService(provider=mock_provider)
    retrieval_service = MemoryRetrievalService(session, emb_service)
    ranking_service = RetrievalRankingService()
    assembler = ContextAssembler()

    # Create a temporary directory for vault export benchmark
    with tempfile.TemporaryDirectory() as tmp_vault_dir:
        export_service = MemoryExportService(session, Path(tmp_vault_dir))

        # Seed data
        print("Seeding SQLite with benchmark data...")
        contact_id, sample_entry, sample_results = seed_data(
            session=session,
            contact_service=contact_service,
            tracking_service=tracking_service,
            ingestion_service=ingestion_service,
            emb_service=emb_service,
        )

        print("Warmup and Benchmarking in progress...")

        # Build runners
        runners = {
            "relationship creation": make_relationship_creation_runner(contact_service),
            "outreach tracking": make_outreach_tracking_runner(tracking_service, contact_id),
            "follow-up recommendation": make_followup_rec_runner(followup_service),
            "memory ingestion": make_memory_ingestion_runner(ingestion_service),
            "significance evaluation": make_significance_evaluation_runner(evaluator),
            "embedding generation": make_embedding_generation_runner(emb_service, sample_entry),
            "semantic retrieval": make_semantic_retrieval_runner(retrieval_service),
            "retrieval reranking": make_retrieval_reranking_runner(
                ranking_service, sample_results
            ),
            "context assembly": make_context_assembly_runner(assembler, sample_results),
            "export synchronization": make_export_sync_runner(export_service),
        }

        # Target performance thresholds (in milliseconds)
        thresholds = {
            "relationship creation": 150.0,
            "outreach tracking": 150.0,
            "follow-up recommendation": 500.0,  # Required: <500ms
            "memory ingestion": 300.0,  # Required: <300ms
            "significance evaluation": 100.0,
            "embedding generation": 200.0,
            "semantic retrieval": 500.0,  # Required: <500ms
            "retrieval reranking": 200.0,  # Required: <200ms
            "context assembly": 200.0,  # Required: <200ms
            "export synchronization": 200.0,
        }

        # Run benchmarks
        results = []
        for name, runner in runners.items():
            print(f"Benchmarking {name}...")
            stats = benchmark_workflow(name, runner, iterations=50, warmup=5)
            results.append(stats)

        # Output results
        print("\n" + "=" * 90)
        print(f"{'PLATFORM WORKFLOW PERFORMANCE BENCHMARKS':^90}")
        print("=" * 90)
        print(
            f"{'Workflow Name':<28} | {'Avg Latency':<12} | {'Max Latency':<12} | "
            f"{'Threshold':<10} | {'Status':<6}"
        )
        print("-" * 90)

        all_passed = True
        for res in results:
            name = res["name"]
            avg = res["avg"]
            max_val = res["max"]
            threshold = thresholds[name]
            status = "PASS" if avg < threshold else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(
                f"{name:<28} | {avg:>9.2f} ms | {max_val:>9.2f} ms | {threshold:>7.1f} ms | {status:<6}"
            )

        print("=" * 90)

        session.close()

        if not all_passed:
            print("\n[ERROR] One or more platform workflows failed to meet target latencies.")
            sys.exit(1)
        else:
            print("\n[SUCCESS] All platform workflows met performance latency criteria.")
            sys.exit(0)


if __name__ == "__main__":
    run_benchmarks()
