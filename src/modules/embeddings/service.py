import gc
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for lazy-loading and interacting with sentence-transformer models.
    Optimized for MacBook Air (low RAM footprint, unloadable model).
    """

    _model_name: str = "all-MiniLM-L6-v2"
    _model: Any = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Any:
        """
        Thread-safe lazy retrieval of the sentence transformer model.
        """
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info("Initializing SentenceTransformer model", model=cls._model_name)
                    cls._model = SentenceTransformer(cls._model_name)
        return cls._model

    @classmethod
    def unload_model(cls) -> None:
        """
        Deregister sentence-transformer model and trigger garbage collection to free RAM.
        """
        with cls._lock:
            if cls._model is not None:
                logger.info("Unloading SentenceTransformer model", model=cls._model_name)
                cls._model = None
                gc.collect()

    def generate_embedding_sync(self, text: str) -> list[float]:
        """
        Synchronously generate embedding for a single text.
        """
        if not text:
            return []
        model = self.get_model()
        vector = model.encode(text, convert_to_numpy=True)
        return [float(x) for x in vector.tolist()]

    def generate_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Synchronously generate embeddings for a batch of texts.
        """
        if not texts:
            return []
        model = self.get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in vector] for vector in vectors.tolist()]

    @staticmethod
    def calculate_similarity(emb1: list[float], emb2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Returns a float between -1.0 and 1.0 (or 0.0 for empty vectors).
        """
        if not emb1 or not emb2:
            return 0.0
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
