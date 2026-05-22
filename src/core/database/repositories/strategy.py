from sqlalchemy.orm import Session

from src.core.database.models import StrategyInsight

from .base import BaseRepository


class StrategyInsightRepository(BaseRepository[StrategyInsight]):
    """
    Data repository for StrategyInsight collections.
    """

    def __init__(self, session: Session):
        super().__init__(StrategyInsight, session)

    def get_by_topic(self, topic: str) -> list[StrategyInsight]:
        """
        Fetch insights matching a specific coaching area.
        """
        return self.session.query(StrategyInsight).filter(StrategyInsight.topic == topic).all()
