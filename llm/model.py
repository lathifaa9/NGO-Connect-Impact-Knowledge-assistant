"""
LLM abstraction layer supporting multiple model backends:
1. OpenAI / LiteLLM API
2. Local Ollama instance
3. Built-in Grounded Extractive Synthesizer (offline, zero-API dependency)
"""

import os
import re
from typing import Any, Dict, List, Optional
import requests

from config.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    UNSUPPORTED_ANSWER_MESSAGE,
)
from llm.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from utils.logger import logger

class GroundedExtractiveSynthesizer:
    """
    Built-in high-precision extractive/abstractive synthesizer.
    Constructs factual, grounded answers directly from retrieved text chunks
    without requiring an external LLM API key. Guarantees 100% adherence to Rule #10.
    """

    def synthesize(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Synthesizes an answer from retrieved chunks."""
        if not retrieved_chunks:
            return UNSUPPORTED_ANSWER_MESSAGE

        # Check if question terms have meaningful matches in the retrieved text
        q_terms = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9]{3,}\b", question)]
        # Filter out common stop words
        stop_words = {"what", "which", "where", "when", "how", "are", "the", "and", "for", "with", "about", "can", "out", "does"}
        filtered_q_terms = [t for t in q_terms if t not in stop_words]

        all_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        if filtered_q_terms and not any(term in all_text for term in filtered_q_terms):
            return UNSUPPORTED_ANSWER_MESSAGE

        # Group key facts by document
        doc_summaries: Dict[str, List[str]] = {}
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Knowledge Base Document")
            text = chunk.get("text", "")

            # Strip the prefix header
            if "]\n" in text:
                text = text.split("]\n", 1)[1]

            # Split into clean paragraphs / bullets
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if doc_name not in doc_summaries:
                doc_summaries[doc_name] = []

            for line in lines:
                if line.startswith("#"):
                    continue
                if line not in doc_summaries[doc_name]:
                    doc_summaries[doc_name].append(line)

        # Build clean synthesized response
        response_parts = []
        response_parts.append(f"Based on the authentic documents in the NGO knowledge base:\n")

        for doc_name, facts in doc_summaries.items():
            response_parts.append(f"#### {doc_name}")
            # Filter and present key facts
            selected_facts = facts[:6]  # top points
            for fact in selected_facts:
                if fact.startswith("-") or fact.startswith("•") or re.match(r"^\d+\.", fact):
                    response_parts.append(f"{fact}")
                elif fact.startswith("|"):
                    # Table row
                    response_parts.append(fact)
                else:
                    response_parts.append(f"- {fact}")
            response_parts.append("")

        return "\n".join(response_parts).strip()


class LLMClient:
    """Unified LLM interface handling API, Ollama, and local fallback synthesis."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
        self.extractive_synthesizer = GroundedExtractiveSynthesizer()

    def generate_response(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_prompt: str = SYSTEM_PROMPT
    ) -> str:
        """
        Generates an answer for the given question using the retrieved chunks.
        """
        if not retrieved_chunks:
            return UNSUPPORTED_ANSWER_MESSAGE

        # Assemble context string
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            context_parts.append(
                f"--- Document Chunk {i} ---\n"
                f"Document: {meta.get('document_name', 'Unknown')}\n"
                f"Category: {meta.get('category', 'General')}\n"
                f"Organization: {meta.get('organization', 'N/A')}\n"
                f"Page: {meta.get('page_number', '1')}\n\n"
                f"{chunk.get('text', '')}\n"
            )
        context_str = "\n".join(context_parts)

        # 1. Try OpenAI if configured
        if self.provider == "openai" and (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
            try:
                from openai import OpenAI
                api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
                user_content = RAG_PROMPT_TEMPLATE.format(
                    context=context_str,
                    question=question,
                    unsupported_message=UNSUPPORTED_ANSWER_MESSAGE
                )
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"OpenAI completion failed: {e}. Falling back to extractive synthesizer.")

        # 2. Try Ollama if configured
        if self.provider == "ollama":
            try:
                user_content = RAG_PROMPT_TEMPLATE.format(
                    context=context_str,
                    question=question,
                    unsupported_message=UNSUPPORTED_ANSWER_MESSAGE
                )
                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": f"{system_prompt}\n\n{user_content}",
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
                res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30)
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
            except Exception as e:
                logger.warning(f"Ollama completion failed: {e}. Falling back to extractive synthesizer.")

        # 3. Default to Grounded Extractive Synthesizer
        return self.extractive_synthesizer.synthesize(question, retrieved_chunks)
