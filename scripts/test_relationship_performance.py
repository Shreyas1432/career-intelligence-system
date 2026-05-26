import statistics
import sys
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.database.models import Base
from src.modules.relationship import (
    CommunicationProfileService,
    ContactCreate,
    ContactService,
    ContactType,
    FollowupRecommendationService,
    OutreachTrackingService,
    RelationshipAnalyticsService,
)


def benchmark_workflow(name: str, fn: Callable[[], Any], iterations: int = 50) -> dict[str, Any]:
    """Helper to run a workflow function multiple times and compile latency stats."""
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


def make_contact_creation_runner(contact_service: ContactService) -> Callable[[], None]:
    """Creates a runner for benchmarking contact creation."""
    contact_index = 0

    def workflow_contact_creation() -> None:
        nonlocal contact_index
        contact_index += 1
        contact_service.create_contact(
            ContactCreate(
                first_name=f"Jane_{contact_index}",
                last_name="Doe",
                company="Google LLC",
                contact_type=ContactType.RECRUITER,
                email=f"jane_{contact_index}@google.com",
            )
        )

    return workflow_contact_creation


def make_outreach_logging_runner(
    tracking_service: OutreachTrackingService, contact_id: UUID
) -> Callable[[], None]:
    """Creates a runner for benchmarking outreach event logging."""
    def workflow_outreach_logging() -> None:
        tracking_service.log_outreach_event(
            contact_id=contact_id,
            method="Email",
            content="Hi Alice, let's connect and sync.",
        )

    return workflow_outreach_logging


def make_state_update_runner(
    tracking_service: OutreachTrackingService, contact_id: UUID
) -> Callable[[], None]:
    """Creates a runner for benchmarking relationship state updates."""
    def workflow_state_updates() -> None:
        tracking_service._sync_relationship_state(contact_id)

    return workflow_state_updates


def make_communication_guidance_runner(
    comm_service: CommunicationProfileService, contact_id: UUID
) -> Callable[[], None]:
    """Creates a runner for benchmarking communication recommendation generation."""
    def workflow_communication_guidance() -> None:
        comm_service.generate_guidance(contact_id)

    return workflow_communication_guidance


def make_followup_rec_runner(
    followup_service: FollowupRecommendationService,
) -> Callable[[], None]:
    """Creates a runner for benchmarking follow-up recommendation generation."""
    def workflow_followup_rec() -> None:
        followup_service.generate_recommendations()

    return workflow_followup_rec


def make_analytics_runner(
    analytics_service: RelationshipAnalyticsService,
) -> Callable[[], None]:
    """Creates a runner for benchmarking analytics aggregation."""
    def workflow_analytics() -> None:
        analytics_service.generate_summary()

    return workflow_analytics


def make_timeline_runner(
    tracking_service: OutreachTrackingService, contact_id: UUID
) -> Callable[[], None]:
    """Creates a runner for benchmarking interaction timeline generation."""
    def workflow_timeline() -> None:
        tracking_service.get_timeline(contact_id)

    return workflow_timeline


def run_benchmarks() -> None:
    """Orchestrates in-memory SQLite performance testing of relationship workflows."""
    print("Initializing in-memory database and services...")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    session = session_local()

    contact_service = ContactService(session)
    tracking_service = OutreachTrackingService(session)
    comm_service = CommunicationProfileService(session)
    followup_service = FollowupRecommendationService(session)
    analytics_service = RelationshipAnalyticsService(session)

    # 1. Benchmark Contact Creation
    print("Benchmarking Contact Creation...")
    workflow_contact_creation = make_contact_creation_runner(contact_service)
    stats_contact = benchmark_workflow("Contact Creation", workflow_contact_creation)

    # Create a base contact to use for downstream single-contact benchmarks
    base_contact = contact_service.create_contact(
        ContactCreate(
            first_name="Alice",
            last_name="Smith",
            company="Stripe Inc",
            contact_type=ContactType.HIRING_MANAGER,
            email="alice@stripe.com",
        )
    )

    # 2. Benchmark Outreach Event Logging
    print("Benchmarking Outreach Event Logging...")
    workflow_outreach_logging = make_outreach_logging_runner(tracking_service, base_contact.id)
    stats_outreach = benchmark_workflow("Outreach Logging", workflow_outreach_logging)

    # 3. Benchmark Relationship State Updates
    print("Benchmarking Relationship State Updates...")
    workflow_state_updates = make_state_update_runner(tracking_service, base_contact.id)
    stats_state = benchmark_workflow("State Updates", workflow_state_updates)

    # 4. Benchmark Communication Recommendation Generation
    print("Benchmarking Communication Recommendation...")
    workflow_comm = make_communication_guidance_runner(comm_service, base_contact.id)
    stats_comm = benchmark_workflow("Communication Rec", workflow_comm)

    # 5. Benchmark Follow-up Recommendation Generation
    print("Benchmarking Follow-up Recommendation...")
    workflow_followup = make_followup_rec_runner(followup_service)
    stats_followup = benchmark_workflow("Follow-up Rec", workflow_followup)

    # 6. Benchmark Analytics Aggregation
    print("Benchmarking Analytics Aggregation...")
    workflow_analytics = make_analytics_runner(analytics_service)
    stats_analytics = benchmark_workflow("Analytics Aggregation", workflow_analytics)

    # 7. Benchmark Interaction Timeline Generation
    print("Benchmarking Interaction Timeline Generation...")
    workflow_timeline = make_timeline_runner(tracking_service, base_contact.id)
    stats_timeline = benchmark_workflow("Timeline Gen", workflow_timeline)

    # Target Performance Thresholds (in ms)
    thresholds = {
        "Contact Creation": 100.0,
        "Outreach Logging": 100.0,
        "State Updates": 100.0,
        "Communication Rec": 300.0,
        "Follow-up Rec": 300.0,
        "Analytics Aggregation": 1000.0,
        "Timeline Gen": 500.0,
    }

    results = [
        stats_contact,
        stats_outreach,
        stats_state,
        stats_comm,
        stats_followup,
        stats_analytics,
        stats_timeline,
    ]

    # Report Benchmark Results
    print("\n" + "=" * 80)
    print(f"{'WORKFLOW PERFORMANCE BENCHMARKS (SQLite In-Memory)':^80}")
    print("=" * 80)
    print(
        f"{'Workflow Name':<25} | {'Avg Latency':<12} | {'Max Latency':<12} | "
        f"{'Threshold':<10} | {'Status':<6}"
    )
    print("-" * 80)

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
            f"{name:<25} | {avg:>9.2f} ms | {max_val:>9.2f} ms | {threshold:>7.1f} ms | {status:<6}"
        )

    print("=" * 80)

    session.close()

    if not all_passed:
        print("\n[ERROR] One or more benchmarks failed to meet performance thresholds.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All workflows passed performance latency checks.")
        sys.exit(0)


if __name__ == "__main__":
    run_benchmarks()
