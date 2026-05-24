from typing import Any

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from src.modules.opportunity_ranking.models import OpportunityRankingResult


class OpportunityRankingRepository:
    """
    Data repository for Opportunity Intelligence Ranking and Scoring results.
    Supports score history, recommendation history, profile comparisons, and recalculations.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_ranking_result(
        self,
        profile_id: int,
        job_id: int,
        ranking_response: Any,
    ) -> OpportunityRankingResult:
        """
        Saves a new ranking result, automatically incrementing run_number for recalculation tracking.
        Supports both Pydantic models and raw dictionary payloads.
        """
        # Convert response to dict if it is a Pydantic model
        if hasattr(ranking_response, "model_dump"):
            data = ranking_response.model_dump()
        elif hasattr(ranking_response, "dict"):
            data = ranking_response.dict()
        else:
            data = ranking_response

        # Determine the next run_number for this profile-job combination
        last_run = (
            self.session.query(OpportunityRankingResult.run_number)
            .filter(
                OpportunityRankingResult.profile_id == profile_id,
                OpportunityRankingResult.job_id == job_id,
            )
            .order_by(desc(OpportunityRankingResult.run_number))
            .first()
        )
        next_run_number = (last_run[0] + 1) if last_run else 1

        db_result = OpportunityRankingResult(
            profile_id=profile_id,
            job_id=job_id,
            overall_score=float(data.get("overall_score", 0.0)),
            recommendation=str(data.get("recommendation", "skip")),
            factor_scores=data.get("factors", {}),
            weights=data.get("weights", {}),
            reasoning_metadata=data.get("reasoning", {}),
            run_number=next_run_number,
        )

        self.session.add(db_result)
        self.session.flush()
        return db_result

    def get_latest_ranking_result(
        self, profile_id: int, job_id: int
    ) -> OpportunityRankingResult | None:
        """
        Gets the latest ranking result for a profile and job comparison.
        """
        return (
            self.session.query(OpportunityRankingResult)
            .filter(
                OpportunityRankingResult.profile_id == profile_id,
                OpportunityRankingResult.job_id == job_id,
            )
            .order_by(desc(OpportunityRankingResult.run_number))
            .first()
        )

    def get_ranking_history(self, profile_id: int, job_id: int) -> list[OpportunityRankingResult]:
        """
        Gets the full recalculation/score history for a profile and job comparison.
        """
        return (
            self.session.query(OpportunityRankingResult)
            .filter(
                OpportunityRankingResult.profile_id == profile_id,
                OpportunityRankingResult.job_id == job_id,
            )
            .order_by(OpportunityRankingResult.run_number)
            .all()
        )

    def get_profile_comparison_history(self, profile_id: int) -> list[OpportunityRankingResult]:
        """
        Gets comparison history of all jobs scored against a profile (latest run per job).
        """
        max_run_subq = (
            self.session.query(
                OpportunityRankingResult.job_id,
                func.max(OpportunityRankingResult.run_number).label("max_run"),
            )
            .filter(OpportunityRankingResult.profile_id == profile_id)
            .group_by(OpportunityRankingResult.job_id)
            .subquery()
        )

        return (
            self.session.query(OpportunityRankingResult)
            .join(
                max_run_subq,
                and_(
                    OpportunityRankingResult.job_id == max_run_subq.c.job_id,
                    OpportunityRankingResult.run_number == max_run_subq.c.max_run,
                ),
            )
            .filter(OpportunityRankingResult.profile_id == profile_id)
            .order_by(desc(OpportunityRankingResult.created_at))
            .all()
        )
