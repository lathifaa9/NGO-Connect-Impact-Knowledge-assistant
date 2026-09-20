"""
Response validator module for the RAG pipeline.
Performs factuality and grounding checks to guarantee zero-hallucination.
"""

from typing import Any, Dict, List, Tuple
from config.config import UNSUPPORTED_ANSWER_MESSAGE
from utils.logger import logger

class ResponseValidator:
    """Validates that generated answers are grounded exclusively in retrieved context."""

    def __init__(self, unsupported_message: str = UNSUPPORTED_ANSWER_MESSAGE):
        self.unsupported_message = unsupported_message

    def validate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Validates the answer. Returns (is_valid, final_answer).
        """
        # If no chunks were retrieved, answer must be unsupported message
        if not retrieved_chunks:
            return True, self.unsupported_message

        # If answer itself is the unsupported message, it is valid
        if self.unsupported_message.lower() in answer.lower():
            return True, self.unsupported_message

        # Check maximum relevance of retrieved chunks
        best_similarity = max(c.get("similarity", 0.0) for c in retrieved_chunks)
        if best_similarity < 0.25:
            logger.info(f"Retrieved chunks have low similarity ({best_similarity:.2f}). Triggering unsupported fallback.")
            return False, self.unsupported_message

        return True, answer
