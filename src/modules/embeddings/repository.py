from sqlalchemy.orm import Session

from src.core.database.models import JobEmbedding, ProfileEmbedding


class EmbeddingRepository:
    """
    Data repository for Job and User Profile embedding persistence.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_job_embedding(
        self, job_id: int, embedding: list[float], session: Session | None = None
    ) -> JobEmbedding:
        """
        Save or update a job embedding.
        """
        sess = session or self.session
        db_emb = sess.query(JobEmbedding).filter(JobEmbedding.job_id == job_id).first()
        if db_emb:
            db_emb.embedding = embedding
        else:
            db_emb = JobEmbedding(job_id=job_id, embedding=embedding)
            sess.add(db_emb)
        sess.flush()
        return db_emb

    def get_job_embedding(self, job_id: int, session: Session | None = None) -> JobEmbedding | None:
        """
        Retrieve a job embedding by job_id.
        """
        sess = session or self.session
        return sess.query(JobEmbedding).filter(JobEmbedding.job_id == job_id).first()

    def save_profile_embedding(
        self, profile_id: int, embedding: list[float], session: Session | None = None
    ) -> ProfileEmbedding:
        """
        Save or update a profile embedding.
        """
        sess = session or self.session
        db_emb = (
            sess.query(ProfileEmbedding).filter(ProfileEmbedding.profile_id == profile_id).first()
        )
        if db_emb:
            db_emb.embedding = embedding
        else:
            db_emb = ProfileEmbedding(profile_id=profile_id, embedding=embedding)
            sess.add(db_emb)
        sess.flush()
        return db_emb

    def get_profile_embedding(
        self, profile_id: int, session: Session | None = None
    ) -> ProfileEmbedding | None:
        """
        Retrieve a profile embedding by profile_id.
        """
        sess = session or self.session
        return (
            sess.query(ProfileEmbedding).filter(ProfileEmbedding.profile_id == profile_id).first()
        )

    def get_all_job_embeddings(
        self, session: Session | None = None
    ) -> list[tuple[int, list[float]]]:
        """
        Retrieve all job embeddings for in-memory similarity comparisons.
        """
        sess = session or self.session
        records = sess.query(JobEmbedding).all()
        return [(r.job_id, r.embedding) for r in records]
