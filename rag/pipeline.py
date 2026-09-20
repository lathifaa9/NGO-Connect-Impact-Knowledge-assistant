"""
End-to-End RAG Pipeline Orchestrator.
Connects Retrieval, Validation, LLM Answer Generation, and Source Citation.
"""

from typing import Any, Dict, List, Optional
from config.config import DEFAULT_TOP_K, UNSUPPORTED_ANSWER_MESSAGE
from llm.model import LLMClient
from rag.response_validator import ResponseValidator
from utils.helpers import format_citations
from utils.logger import logger
from vectorstore.retriever import NGORetriever

class NGORAGPipeline:
    """Orchestrates the question answering workflow with strict source grounding."""

    def __init__(
        self,
        retriever: Optional[NGORetriever] = None,
        llm_client: Optional[LLMClient] = None,
        validator: Optional[ResponseValidator] = None
    ):
        self.retriever = retriever or NGORetriever()
        self.llm_client = llm_client or LLMClient()
        self.validator = validator or ResponseValidator()

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        category: Optional[str] = None,
        focus_area: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a user question.
        Returns a dictionary containing the answer, citations, and retrieved chunks.
        """
        logger.info(f"Processing query: '{question}' (Category: {category}, Focus: {focus_area})")

        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            category=category,
            focus_area=focus_area
        )

        if not chunks:
            logger.info("No matching chunks found above threshold.")
            return {
                "question": question,
                "answer": UNSUPPORTED_ANSWER_MESSAGE,
                "citations": "",
                "chunks": [],
                "is_grounded": True,
                "is_supported": False
            }

        # 2. Generate answer via LLM
        raw_answer = self.llm_client.generate_response(question, chunks)

        # 3. Validate response
        is_valid, validated_answer = self.validator.validate(question, raw_answer, chunks)

        # 4. Generate Citations if supported
        citations = ""
        is_supported = validated_answer.strip() != UNSUPPORTED_ANSWER_MESSAGE
        if is_supported:
            citations = format_citations(chunks)

        full_answer = f"{validated_answer}{citations}" if citations else validated_answer

        return {
            "question": question,
            "answer": full_answer,
            "raw_answer": validated_answer,
            "citations": citations,
            "chunks": chunks,
            "is_grounded": is_valid,
            "is_supported": is_supported
        }
