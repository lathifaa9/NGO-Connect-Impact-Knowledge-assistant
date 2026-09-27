"""
Hybrid Retriever module.
Combines semantic vector search, metadata filtering (by category/focus area),
and relevance scoring to retrieve the most contextually pertinent chunks.
"""

from typing import Any, Dict, List, Optional
from config.config import DEFAULT_TOP_K, MAX_COSINE_DISTANCE
from vectorstore.chroma_db import ChromaVectorStore
from utils.logger import logger

class NGORetriever:
    """Retriever for the NGO knowledge base with metadata filtering."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def build_filter(
        self,
        category: Optional[str] = None,
        focus_area: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Constructs a ChromaDB-compliant filter dictionary."""
        conditions = []
        if category and category != "All Categories":
            conditions.append({"category": {"$eq": category}})
        if focus_area and focus_area != "All Focus Areas":
            conditions.append({"focus_area": {"$eq": focus_area}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category: Optional[str] = None,
        focus_area: Optional[str] = None,
        max_distance: float = MAX_COSINE_DISTANCE
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching query and optional filters.
        Filters out low-similarity chunks based on max_distance.
        """
        where_filter = self.build_filter(category, focus_area)

        # Retrieve a broader candidate pool to avoid missing relevant items
        candidate_k = min(15, max(top_k * 3, 10))
        raw_results = self.vector_store.query(
            query_text=query,
            n_results=candidate_k,
            where_filter=where_filter
        )

        retrieved: List[Dict[str, Any]] = []
        if not raw_results or "documents" not in raw_results or not raw_results["documents"]:
            return retrieved

        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0] if "metadatas" in raw_results else [{}] * len(docs)
        distances = raw_results["distances"][0] if "distances" in raw_results else [0.0] * len(docs)
        ids = raw_results["ids"][0] if "ids" in raw_results else [""] * len(docs)

        stop_terms = {
            "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
            "are", "was", "were", "been", "being", "have", "has", "had", "does",
            "did", "will", "would", "shall", "should", "could", "might", "must",
            "can", "the", "and", "for", "with", "about", "out", "any", "some",
            "tell", "give", "show", "explain", "describe", "detail", "details",
            "please", "name", "list", "about", "they", "them", "their", "from",
            "work", "works"
        }
        query_terms = [t for t in query.lower().split() if len(t) > 3 and t not in stop_terms]

        has_any_content_match = False
        candidates: List[Dict[str, Any]] = []

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Strict relevance filter: reject if cosine distance is too high
            if dist > max_distance:
                continue

            similarity = max(0.0, 1.0 - dist)
            doc_lower = doc_text.lower()
            meta_str = (
                meta.get("document_name", "") + " " +
                meta.get("focus_area", "") + " " +
                meta.get("category", "")
            ).lower()

            # Measure overlap with document text and metadata
            text_matches = sum(
                1 for term in query_terms
                if term in doc_lower or (len(term) > 3 and term.rstrip("s") in doc_lower)
            )
            meta_matches = sum(
                1 for term in query_terms
                if term in meta_str or (len(term) > 3 and term.rstrip("s") in meta_str)
            )

            is_matched = (text_matches > 0 or meta_matches > 0)
            if is_matched:
                has_any_content_match = True

            boosted_score = similarity + (0.03 * min(text_matches, 4)) + (0.08 * min(meta_matches, 3))

            candidates.append({
                "id": chunk_id,
                "text": doc_text,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity,
                "score": boosted_score,
                "is_matched": is_matched
            })

        # If specific content matches were found, prune distant chunks lacking relevant terms
        if has_any_content_match:
            candidates = [c for c in candidates if c["is_matched"] or c["distance"] < 0.40]

        # Sort by boosted score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

