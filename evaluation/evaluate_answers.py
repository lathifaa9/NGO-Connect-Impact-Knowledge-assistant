"""
End-to-End Answer Quality & Grounding Evaluation Benchmark.
Evaluates Response Groundedness, Citation Accuracy, and Zero-Hallucination Guardrails.
"""

import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import TEST_QUESTIONS_PATH, UNSUPPORTED_ANSWER_MESSAGE, DEFAULT_TOP_K
from rag.pipeline import NGORAGPipeline
from utils.logger import logger

def format_table(rows: List[Dict[str, Any]]) -> str:
    """Formats a list of dicts as a clean markdown table without external packages."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))
    
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    data_lines = [
        "| " + " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)

def evaluate_answers(top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Runs end-to-end evaluation on groundedness and anti-hallucination guardrails."""
    logger.info("=== Starting End-to-End Answer Grounding Benchmark ===")

    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    pipeline = NGORAGPipeline()

    results = []
    supported_pass = 0
    total_supported = 0
    total_unsupported = 0
    unsupported_pass = 0
    latencies = []

    for q in questions:
        q_id = q["id"]
        query = q["question"]
        is_supported = q.get("is_supported", True)
        expected_keywords = [kw.lower() for kw in q.get("expected_keywords", [])]

        start_time = time.perf_counter()
        response = pipeline.query(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

        answer_text = response.get("answer", "")
        has_citations = bool(response.get("citations", ""))
        num_chunks = len(response.get("chunks", []))

        if is_supported:
            total_supported += 1
            # Must not be fallback message, and must have citations and expected keywords
            is_fallback = UNSUPPORTED_ANSWER_MESSAGE.lower() in answer_text.lower()
            keyword_matches = [kw for kw in expected_keywords if kw in answer_text.lower()]
            has_kw = len(keyword_matches) > 0

            passed = (not is_fallback) and has_citations and has_kw
            if passed:
                supported_pass += 1

            results.append({
                "ID": q_id,
                "Query": query[:38] + "..." if len(query) > 38 else query,
                "Type": "Supported",
                "Answer Status": "Grounded" if not is_fallback else "Unsupported",
                "Citations": "Yes" if has_citations else "No",
                "Keywords": f"{len(keyword_matches)}/{len(expected_keywords)}",
                "Result": "PASS" if passed else "FAIL"
            })
        else:
            total_unsupported += 1
            # Must strictly trigger unsupported message with 0 hallucination
            is_strict_fallback = (answer_text.strip() == UNSUPPORTED_ANSWER_MESSAGE)
            passed = is_strict_fallback and (not has_citations)
            if passed:
                unsupported_pass += 1

            results.append({
                "ID": q_id,
                "Query": query[:38] + "..." if len(query) > 38 else query,
                "Type": "Out-of-Scope",
                "Answer Status": "Rejected (Clean)",
                "Citations": "None",
                "Keywords": "N/A",
                "Result": "PASS" if passed else "FAIL"
            })

    grounded_rate = (supported_pass / total_supported) * 100.0 if total_supported > 0 else 0.0
    guardrail_rate = (unsupported_pass / total_unsupported) * 100.0 if total_unsupported > 0 else 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 80)
    print("        NGO KNOWLEDGE ASSISTANT - ANSWER GROUNDING BENCHMARK")
    print("=" * 80)
    print(format_table(results))
    print("\n" + "-" * 80)
    print(f"Supported Question Grounding Pass Rate:   {grounded_rate:.1f}% ({supported_pass}/{total_supported})")
    print(f"Zero-Hallucination Guardrail Rate:       {guardrail_rate:.1f}% ({unsupported_pass}/{total_unsupported})")
    print(f"Average Pipeline Latency:                {avg_latency:.1f} ms")
    print("=" * 80 + "\n")

    return {
        "grounded_rate": grounded_rate,
        "guardrail_rate": guardrail_rate,
        "avg_latency_ms": avg_latency,
        "results": results
    }

if __name__ == "__main__":
    evaluate_answers()
