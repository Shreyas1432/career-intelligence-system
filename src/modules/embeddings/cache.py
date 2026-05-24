
from src.core.cache.manager import CacheManager


class EmbeddingCache:
    """
    Wrapper around CacheManager specifically for all-MiniLM-L6-v2 embeddings.
    """

    def __init__(self, manager: CacheManager | None = None) -> None:
        self.manager = manager or CacheManager()
        self.model_name = "all-MiniLM-L6-v2"

    def get_cached_embedding(self, text: str) -> list[float] | None:
        """
        Retrieve a cached embedding from SQLite cache store.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        val = self.manager.get(key)
        if val is not None and isinstance(val, list):
            return [float(x) for x in val]
        return None

    def set_cached_embedding(self, text: str, embedding: list[float]) -> None:
        """
        Store a generated embedding in SQLite cache store.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        self.manager.set(key, "embedding", embedding)

    def get_cached_embeddings_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """
        Retrieve cached embeddings for a list of texts in batch.
        """
        results = {}
        for text in texts:
            emb = self.get_cached_embedding(text)
            if emb is not None:
                results[text] = emb
        return results

    # Async variants to prevent blocking the event loop
    async def get_cached_embedding_async(self, text: str) -> list[float] | None:
        """
        Asynchronously retrieve a cached embedding.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        val = await self.manager.get_async(key)
        if val is not None and isinstance(val, list):
            return [float(x) for x in val]
        return None

    async def set_cached_embedding_async(self, text: str, embedding: list[float]) -> None:
        """
        Asynchronously store a generated embedding in cache.
        """
        key = self.manager.generate_embedding_key(self.model_name, text)
        await self.manager.set_async(key, "embedding", embedding)

    async def get_cached_embeddings_batch_async(self, texts: list[str]) -> dict[str, list[float]]:
        """
        Asynchronously retrieve cached embeddings for a batch of texts.
        """
        results = {}
        for text in texts:
            emb = await self.get_cached_embedding_async(text)
            if emb is not None:
                results[text] = emb
        return results
