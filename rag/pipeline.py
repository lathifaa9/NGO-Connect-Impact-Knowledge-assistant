"""
End-to-End RAG Pipeline Orchestrator.
Connects Retrieval, Validation, LLM Answer Generation, and Source Citation.
"""

import re
from typing import Any, Dict, List, Optional

from config.config import DEFAULT_TOP_K, UNSUPPORTED_ANSWER_MESSAGE
from llm.model import LLMClient
from rag.response_validator import ResponseValidator
from utils.helpers import format_citations
from utils.logger import logger
from vectorstore.retriever import NGORetriever


class NGORAGPipeline:
    """Orchestrates the question answering workflow with strict source grounding."""

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|heya|howdy|hola|greetings)\b",
        r"^good\s+(morning|afternoon|evening|day)\b",
        r"^(namaste|namaskar|vanakkam|pranam)\b",
        r"^how\s+are\s+you\b",
    ]

    CAPABILITY_PATTERNS = [
        r"^(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+this|help|how\s+do\s+you\s+work)\b",
        r"^(tell\s+me\s+about\s+yourself|what\s+do\s+you\s+know)\b",
    ]

    COURTESY_PATTERNS = [
        r"^(thank\s+you|thanks|thank\s+u|thx)\b",
        r"^(bye|goodbye|see\s+you|cya)\b",
    ]

    def __init__(
        self,
        retriever: Optional[NGORetriever] = None,
        llm_client: Optional[LLMClient] = None,
        validator: Optional[ResponseValidator] = None
    ):
        self.retriever = retriever or NGORetriever()
        self.llm_client = llm_client or LLMClient()
        self.validator = validator or ResponseValidator()

    def _check_conversational_intent(self, question: str) -> Optional[Dict[str, Any]]:
        """Identifies pleasantries, capabilities questions, or greetings without triggering vector search."""
        clean_q = question.strip().lower()
        # Remove trailing punctuation
        clean_q = re.sub(r"[?!.,;]+$", "", clean_q).strip()

        # 1. Greetings
        for pat in self.GREETING_PATTERNS:
            if re.match(pat, clean_q, re.IGNORECASE):
                ans = (
                    "Hello! Welcome to the **NGO Connect & Impact Knowledge Assistant**.\n\n"
                    "I am an AI assistant grounded in authentic Indian statutory frameworks, NITI Aayog guidelines, "
                    "FCRA regulations, CSR policies, and verified NGO impact reports.\n\n"
                    "**You can ask me questions such as:**\n"
                    "- *How are NGOs registered in India under Trust, Society, or Section 8?*\n"
                    "- *What are the mandatory compliance rules under FCRA 2020?*\n"
                    "- *What are the CSR funding guidelines under Section 135?*\n"
                    "- *Which government grant schemes are available for rural development?*\n"
                    "- *What audited impact outcomes were reported by Goonj or Akshaya Patra?*\n\n"
                    "How can I assist you with NGO operations, frameworks, or impact data today?"
                )
                return {
                    "question": question,
                    "answer": ans,
                    "raw_answer": ans,
                    "citations": "",
                    "chunks": [],
                    "is_grounded": True,
                    "is_supported": True,
                    "is_greeting": True
                }

        # 2. Capabilities / Identity
        for pat in self.CAPABILITY_PATTERNS:
            if re.match(pat, clean_q, re.IGNORECASE):
                ans = (
                    "I am the **NGO Connect & Impact Knowledge Assistant**, designed for the B.Tech CSE Final Year Capstone Project (**DATA CODEX — TEAM 10**).\n\n"
                    "**Core Capabilities:**\n"
                    "1. **NGO Formation & Legal Structures**: Detail statutory rules for Public Charitable Trusts, Societies, and Section 8 Companies.\n"
                    "2. **Statutory Compliance**: Explain FCRA 2020 requirements, NITI Aayog NGO Darpan registration, Section 12A/80G tax exemptions, and Section 135 CSR regulations.\n"
                    "3. **Government Scheme Implementation**: Detail partnership frameworks with DAY-NRLM (MoRD), DDRS (MSJE), and other central schemes.\n"
                    "4. **Verified Programmatic Impact**: Provide authentic impact statistics from verified NGO annual and impact reports.\n"
                    "5. **Zero Hallucination Guardrail**: I answer solely from verified documents. If information is not in the knowledge base, I decline to answer rather than speculate."
                )
                return {
                    "question": question,
                    "answer": ans,
                    "raw_answer": ans,
                    "citations": "",
                    "chunks": [],
                    "is_grounded": True,
                    "is_supported": True,
                    "is_greeting": True
                }

        # 3. Courtesy / Farewells
        for pat in self.COURTESY_PATTERNS:
            if re.match(pat, clean_q, re.IGNORECASE):
                ans = (
                    "You're very welcome! If you have any further questions about NGO registration, compliance, "
                    "schemes, or audited impact outcomes, feel free to ask anytime. Wishing you the best in your community work!"
                )
                return {
                    "question": question,
                    "answer": ans,
                    "raw_answer": ans,
                    "citations": "",
                    "chunks": [],
                    "is_grounded": True,
                    "is_supported": True,
                    "is_greeting": True
                }

        return None

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

        # Check for conversational pleasantries/greetings first
        conversational_res = self._check_conversational_intent(question)
        if conversational_res is not None:
            return conversational_res

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
                "raw_answer": UNSUPPORTED_ANSWER_MESSAGE,
                "citations": "",
                "chunks": [],
                "is_grounded": True,
                "is_supported": False,
                "is_greeting": False
            }

        # 2. Generate answer via LLM / Synthesizer
        raw_answer = self.llm_client.generate_response(question, chunks)

        # 3. Validate response
        is_valid, validated_answer = self.validator.validate(question, raw_answer, chunks)

        # 4. Generate Citations only if answer is supported and strictly grounded
        norm_answer = validated_answer.replace("’", "'").strip().lower()
        norm_unsupported = UNSUPPORTED_ANSWER_MESSAGE.replace("’", "'").strip().lower()
        
        is_supported = is_valid and (norm_answer != norm_unsupported) and ("not have enough information" not in norm_answer)

        citations = ""
        final_chunks = []
        if is_supported:
            citations = format_citations(chunks)
            final_chunks = chunks
        else:
            validated_answer = UNSUPPORTED_ANSWER_MESSAGE
            final_chunks = []
            citations = ""

        full_answer = f"{validated_answer}{citations}" if citations else validated_answer

        return {
            "question": question,
            "answer": full_answer,
            "raw_answer": validated_answer,
            "citations": citations,
            "chunks": final_chunks,
            "is_grounded": is_valid,
            "is_supported": is_supported,
            "is_greeting": False
        }

