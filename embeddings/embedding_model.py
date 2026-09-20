"""
Embedding model wrapper using SentenceTransformers.
Computes dense vector representations for queries and document chunks.
"""

from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE
from utils.logger import logger

class EmbeddingModel:
    """Wrapper around SentenceTransformer with device management and batching."""

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingModel":
        """Singleton accessor to prevent reloading weights into memory repeatedly."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def embed_texts(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
        """Generates dense vector embeddings for a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates a normalized embedding for a single user query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()
