"""
LLM abstraction layer supporting multiple model backends:
1. OpenAI / LiteLLM API
2. Local Ollama instance
3. Built-in Grounded Extractive Synthesizer (offline, zero-API dependency)
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
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

    @staticmethod
    def _is_valid_fact_line(line: str) -> bool:
        """Filters out fragmented lines, table borders, and truncated sentence pieces."""
        line = line.strip()
        if not line or len(line) < 18:
            return False
        if line.startswith(("#", "---", "===", "***", "___", "```")):
            return False
        # If line is a bullet or number, inspect content after prefix
        content = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        if not content or len(content) < 15:
            return False
        # Must begin with an uppercase character, digit, or quotation/bracket
        first_char = content[0]
        if not (first_char.isupper() or first_char.isdigit() or first_char in ['"', "'", "“", "”", "(", "[", "—"]):
            return False
        # Filter out common truncation artifacts
        lower_content = content.lower()
        if lower_content.startswith(("overning ", "ative ", "is the year", "and other ", "etc.")):
            return False
        return True

    def synthesize(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Synthesizes an answer from retrieved chunks."""
        if not retrieved_chunks:
            return UNSUPPORTED_ANSWER_MESSAGE

        # Validate similarity threshold
        best_sim = max((c.get("similarity", 0.0) for c in retrieved_chunks), default=0.0)
        if best_sim < 0.42:
            return UNSUPPORTED_ANSWER_MESSAGE

        # Check if question terms have meaningful matches in the retrieved text
        q_terms = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9]{3,}\b", question)]
        # Filter out common stop/question words
        stop_words = {
            "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
            "are", "was", "were", "been", "being", "have", "has", "had", "does",
            "did", "will", "would", "shall", "should", "could", "might", "must",
            "can", "the", "and", "for", "with", "about", "out", "any", "some",
            "tell", "give", "show", "explain", "describe", "detail", "details",
            "please", "name", "list", "available", "information"
        }
        filtered_q_terms = [t for t in q_terms if t not in stop_words]

        all_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        if filtered_q_terms:
            matches = sum(
                1 for term in filtered_q_terms
                if term in all_text or (len(term) > 3 and term.rstrip("s") in all_text)
            )
            if matches == 0:
                return UNSUPPORTED_ANSWER_MESSAGE

        # Group key facts by document
        doc_summaries: Dict[str, List[str]] = {}
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Knowledge Base Document")
            text = chunk.get("text", "")

            # Strip the prefix header if present
            if "]\n" in text:
                text = text.split("]\n", 1)[1]

            # Split into clean lines
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if doc_name not in doc_summaries:
                doc_summaries[doc_name] = []

            for line in lines:
                if self._is_valid_fact_line(line):
                    clean_line = re.sub(r"^[-*•\s]+", "", line).strip()
                    if clean_line not in doc_summaries[doc_name]:
                        doc_summaries[doc_name].append(clean_line)

        if not doc_summaries:
            return UNSUPPORTED_ANSWER_MESSAGE

        def score_fact(fact_text: str) -> float:
            f_lower = fact_text.lower()
            score = 0.0
            for term in filtered_q_terms:
                if term in f_lower or (len(term) > 3 and term.rstrip("s") in f_lower):
                    score += 2.0
            if any(k in f_lower for k in ["defined as", "is a", "refers to", "requirements", "eligibility", "mandate", "registered under", "act", "percent", "%", "crore", "lakh", "guidelines"]):
                score += 1.0
            if fact_text.endswith(":") or len(fact_text) < 25:
                score -= 1.5
            return score

        # Rank facts within each document
        scored_docs: List[Tuple[float, str, List[str]]] = []
        for doc_name, facts in doc_summaries.items():
            if not facts:
                continue
            scored_facts = [(score_fact(f), f) for f in facts]
            scored_facts.sort(key=lambda x: x[0], reverse=True)
            max_s = scored_facts[0][0] if scored_facts else 0.0
            top_doc_facts = [f for _, f in scored_facts[:3]]
            scored_docs.append((max_s, doc_name, top_doc_facts))

        # Sort documents by relevance
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Build clean, direct synthesized response
        response_parts = ["Based on the verified NGO knowledge base:\n"]
        total_points = 0
        max_total_points = 5

        # Shorten document names for clean inline attribution if multiple
        multiple_docs = len(scored_docs) > 1

        for _, doc_name, facts in scored_docs:
            if total_points >= max_total_points:
                break
            short_name = doc_name.split(" - ")[0].strip()
            short_name = re.sub(r"\s*\([^)]*\)", "", short_name).strip()

            facts_to_take = facts[:2] if multiple_docs else facts[:4]
            for fact in facts_to_take:
                if total_points >= max_total_points:
                    break
                if multiple_docs:
                    response_parts.append(f"- **{short_name}**: {fact}")
                else:
                    response_parts.append(f"- {fact}")
                total_points += 1

        return "\n".join(response_parts).strip()


class LLMClient:
    """Unified LLM interface handling Groq, OpenAI, Ollama, and local fallback synthesis."""

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

        # 1. Try Groq if GROQ_API_KEY is available
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=groq_key)
                user_content = RAG_PROMPT_TEMPLATE.format(
                    context=context_str,
                    question=question,
                    unsupported_message=UNSUPPORTED_ANSWER_MESSAGE
                )
                chat_res = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                ans = chat_res.choices[0].message.content.strip()
                if ans:
                    return ans
            except Exception as e:
                logger.warning(f"Groq API call failed: {e}. Falling back to extractive synthesizer.")

        # 2. Try OpenAI if configured
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

        # 3. Try Ollama if configured
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

        # 4. Built-in Grounded Extractive Synthesizer
        return self.extractive_synthesizer.synthesize(question, retrieved_chunks)

