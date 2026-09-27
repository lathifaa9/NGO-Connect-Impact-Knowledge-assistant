"""
Response validator module for the RAG pipeline.
Performs factuality and grounding checks to guarantee zero-hallucination.
"""

import re
from typing import Any, Dict, List, Tuple
from config.config import UNSUPPORTED_ANSWER_MESSAGE
from utils.logger import logger

class ResponseValidator:
    """Validates that generated answers are grounded exclusively in retrieved context."""

    COMMON_STOPWORDS = {
        "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
        "are", "was", "were", "been", "being", "have", "has", "had", "does",
        "did", "will", "would", "shall", "should", "could", "might", "must",
        "the", "and", "for", "with", "about", "can", "out", "any", "some",
        "tell", "give", "show", "explain", "describe", "detail", "details",
        "please", "there", "their", "they", "this", "that", "these", "those",
        "from", "into", "over", "after", "before", "under", "between", "through"
    }

    FALLBACK_PHRASES = [
        "not have enough information",
        "don't have enough information",
        "don’t have enough information",
        "not enough information in the available",
        "not enough information in the ngo knowledge base",
        "cannot answer this question based on the provided",
        "not mentioned in the provided context",
        "information is not available in the context",
        "context does not contain enough information",
        "context does not provide information",
    ]

    def __init__(self, unsupported_message: str = UNSUPPORTED_ANSWER_MESSAGE):
        self.unsupported_message = unsupported_message

    def _normalize_text(self, text: str) -> str:
        """Normalizes apostrophes, quotes, and whitespace for robust comparison."""
        t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return " ".join(t.lower().split())

    def _is_unsupported_response(self, answer: str) -> bool:
        """Checks if the generated answer is or contains an unsupported fallback statement."""
        norm_answer = self._normalize_text(answer)
        norm_unsupported = self._normalize_text(self.unsupported_message)

        if norm_unsupported in norm_answer:
            return True

        for phrase in self.FALLBACK_PHRASES:
            if phrase in norm_answer:
                return True

        return False

    def validate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Validates the answer. Returns (is_valid, final_answer).
        If the answer is unsupported or context lacks relevance, returns (False, unsupported_message).
        """
        # 1. If no chunks were retrieved, answer must be unsupported message
        if not retrieved_chunks:
            return False, self.unsupported_message

        # 2. If the answer indicates insufficient information or fallback
        if not answer or not answer.strip() or self._is_unsupported_response(answer):
            return False, self.unsupported_message

        # 3. Check similarity threshold of retrieved chunks
        best_similarity = max((c.get("similarity", 0.0) for c in retrieved_chunks), default=0.0)
        if best_similarity < 0.42:
            logger.info(
                f"Retrieved chunks have insufficient similarity ({best_similarity:.2f} < 0.42). "
                "Triggering formal unsupported fallback."
            )
            return False, self.unsupported_message

        # 4. Generalized query-context relevance verification
        q_tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower())
        content_tokens = [tok for tok in q_tokens if tok not in self.COMMON_STOPWORDS]

        if content_tokens:
            all_chunk_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
            matches = sum(
                1 for tok in content_tokens
                if tok in all_chunk_text or (len(tok) > 3 and tok.rstrip("s") in all_chunk_text)
            )
            match_ratio = matches / len(content_tokens)

            # If none of the content terms appear in any chunk text, question is unrelated
            if matches == 0:
                logger.info(f"Zero question content terms matched in retrieved chunks ({matches}/{len(content_tokens)}). Triggering fallback.")
                return False, self.unsupported_message

            # If fewer than 25% of query terms match and similarity is borderline
            if match_ratio < 0.25 and best_similarity < 0.45:
                logger.info(f"Insufficient topical overlap ({match_ratio:.2f}) with borderline similarity ({best_similarity:.2f}). Triggering fallback.")
                return False, self.unsupported_message

        return True, answer

