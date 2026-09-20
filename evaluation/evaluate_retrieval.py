"""
Retrieval Evaluation Benchmark Script.
Evaluates Precision@K, Hit Rate, and MRR on the benchmark test questions.
"""

import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import TEST_QUESTIONS_PATH, DEFAULT_TOP_K
from vectorstore.retriever import NGORetriever
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

def evaluate_retrieval(top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Runs retrieval evaluation against the test questions suite."""
    logger.info(f"=== Starting Retrieval Evaluation Benchmark (Top-K={top_k}) ===")
    
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = NGORetriever()

    results = []
    hits = 0
    total_supported = 0
    total_unsupported = 0
    unsupported_correct = 0
    rr_scores = []
    latencies = []

    for q in questions:
        q_id = q["id"]
        query = q["question"]
        is_supported = q.get("is_supported", True)
        expected_keywords = [kw.lower() for kw in q.get("expected_keywords", [])]

        start_time = time.perf_counter()
        retrieved = retriever.retrieve(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

        if is_supported:
            total_supported += 1
            # Check if any chunk matches expected keywords
            hit = False
            reciprocal_rank = 0.0

            for rank, chunk in enumerate(retrieved, start=1):
                chunk_text = (chunk["text"] + " " + chunk.get("metadata", {}).get("document_name", "")).lower()
                matches = [kw for kw in expected_keywords if kw in chunk_text]
                if matches:
                    if not hit:
                        hit = True
                        reciprocal_rank = 1.0 / rank
                    break

            if hit:
                hits += 1
            rr_scores.append(reciprocal_rank)

            top_doc = retrieved[0]["metadata"].get("document_name", "N/A") if retrieved else "None"
            best_sim = retrieved[0]["similarity"] if retrieved else 0.0

            results.append({
                "ID": q_id,
                "Query": query[:42] + "..." if len(query) > 42 else query,
                "Supported": "Yes",
                "Hit": "PASS" if hit else "FAIL",
                "Best Sim": f"{best_sim:.3f}",
                "Top Retrieved Document": top_doc[:38]
            })
        else:
            total_unsupported += 1
            best_sim = retrieved[0]["similarity"] if retrieved else 0.0
            is_clean = len(retrieved) == 0 or best_sim < 0.25
            if is_clean:
                unsupported_correct += 1

            results.append({
                "ID": q_id,
                "Query": query[:42] + "..." if len(query) > 42 else query,
                "Supported": "No (Guardrail)",
                "Hit": "PASS" if is_clean else "FAIL",
                "Best Sim": f"{best_sim:.3f}",
                "Top Retrieved Document": "Filtered Out" if not retrieved else retrieved[0]["metadata"].get("document_name", "N/A")[:38]
            })

    hit_rate = (hits / total_supported) * 100.0 if total_supported > 0 else 0.0
    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    guardrail_acc = (unsupported_correct / total_unsupported) * 100.0 if total_unsupported > 0 else 100.0

    print("\n" + "=" * 80)
    print("           NGO KNOWLEDGE ASSISTANT - RETRIEVAL BENCHMARK RESULTS")
    print("=" * 80)
    print(format_table(results))
    print("\n" + "-" * 80)
    print(f"Supported Questions Hit Rate @ {top_k}: {hit_rate:.1f}% ({hits}/{total_supported})")
    print(f"Mean Reciprocal Rank (MRR):          {mrr:.3f}")
    print(f"Out-of-Scope Guardrail Accuracy:     {guardrail_acc:.1f}% ({unsupported_correct}/{total_unsupported})")
    print(f"Average Retrieval Latency:           {avg_latency:.1f} ms")
    print("=" * 80 + "\n")

    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "guardrail_accuracy": guardrail_acc,
        "avg_latency_ms": avg_latency,
        "total_evaluated": len(questions),
        "results": results
    }

if __name__ == "__main__":
    evaluate_retrieval()
