from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database.models import Base


class OpportunityRankingResult(Base):
    """
    SQLAlchemy model storing historical opportunity intelligence ranking and scoring results.
    Supports score history, recommendation history, and recalculation tracking.
    """

    __tablename__ = "opportunity_ranking_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("user_profile.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("job_intelligence.id", ondelete="CASCADE"), nullable=False, index=True
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(50), nullable=False)
    factor_scores: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    weights: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    reasoning_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    run_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<OpportunityRankingResult profile_id={self.profile_id} job_id={self.job_id} "
            f"run={self.run_number} score={self.overall_score}>"
        )
