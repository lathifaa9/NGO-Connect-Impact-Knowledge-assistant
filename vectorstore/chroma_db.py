"""
ChromaDB persistent vector store integration.
Manages collection creation, chunk upsert with embeddings, and filtered queries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from config.config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from embeddings.embedding_model import EmbeddingModel
from ingestion.metadata import DocumentChunk
from utils.logger import logger

class ChromaVectorStore:
    """Persistent Chroma vector database client for NGO knowledge chunks."""

    def __init__(self, persist_dir: Path = CHROMA_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel.get_instance()

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to ChromaDB at '{self.persist_dir}', collection: '{collection_name}', items: {self.collection.count()}")

    def count(self) -> int:
        """Returns the total number of document chunks in the collection."""
        return self.collection.count()

    def upsert_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """
        Computes embeddings and upserts document chunks into ChromaDB in batches.
        """
        if not chunks:
            return 0

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            embeddings = self.embedding_model.embed_texts(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_upserted += len(batch)

        logger.info(f"Upserted {total_upserted} chunks into ChromaDB collection '{self.collection_name}'.")
        return total_upserted

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a vector search with optional metadata filtering.
        """
        query_embedding = self.embedding_model.embed_query(query_text)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)
        return results

    def reset(self) -> None:
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'.")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
