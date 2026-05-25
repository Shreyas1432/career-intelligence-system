"""
Example script demonstrating usage of the Database layer and Repository pattern.
To run this:
    uv run python examples/database_usage.py
"""

from src.core.database import Application, Job, get_db_session, init_db
from src.core.database.repositories import ApplicationRepository, JobRepository


def main() -> None:
    print("Initializing local SQLite database tables...")
    init_db()

    # Establish database session context
    with get_db_session() as session:
        job_repo = JobRepository(session)
        app_repo = ApplicationRepository(session)

        print("\nCreating Job entry...")
        job = Job(
            title="Senior AI Engineer",
            company="Innovative AI Solutions",
            location="San Francisco, CA (Hybrid)",
            salary_range="$180k - $220k",
            url="https://innovative-ai.example.com/careers/senior-ai",
        )
        job_repo.create(job)
        # Flush or commit session updates to trigger SQL generation and fetch PKs
        session.flush()
        print(f"Created Job ID: {job.id} -> '{job.title}'")

        print("\nLinking Application to Job...")
        app = Application(
            job_id=job.id,
            status="Interviewing",
            notes="Completed screening call with Recruiting Lead.",
            resume_version="v2.1_Lead",
        )
        app_repo.create(app)
        session.flush()
        print(f"Created Application ID: {app.id} linked to Job: {app.job_id}")

    # Verify updates in a separate transaction context
    with get_db_session() as session:
        job_repo = JobRepository(session)
        app_repo = ApplicationRepository(session)

        print("\nSearching target jobs by keyword 'AI'...")
        jobs = job_repo.search_jobs("AI")
        for j in jobs:
            print(f"- {j.title} at {j.company} (ID: {j.id})")

        print("\nFetching applications marked as 'Interviewing'...")
        apps = app_repo.get_by_status("Interviewing")
        for a in apps:
            # Use or_raise: the ID is known-valid from get_by_status; missing = data error.
            detailed_app = app_repo.get_application_with_details_or_raise(a.id)
            print(f"- Job: {detailed_app.job.title} at {detailed_app.job.company}")
            print(f"  Status: {detailed_app.status}")
            print(f"  Notes: {detailed_app.notes}")


if __name__ == "__main__":
    main()
