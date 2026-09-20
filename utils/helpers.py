"""
Helper utilities for hashing, citations formatting, and file operations.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union

def compute_hash(text: str) -> str:
    """Computes a stable SHA-256 hash string for text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def save_json(data: Union[Dict, List], file_path: Union[str, Path], indent: int = 2) -> None:
    """Saves a Python dictionary or list to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(file_path: Union[str, Path]) -> Union[Dict, List, None]:
    """Loads JSON data from file, returning None if not found."""
    path = Path(file_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_citations(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a clean, standardized Markdown citations section from retrieved chunks.
    Groups citations by document name to avoid duplicate source listings.
    """
    if not retrieved_chunks:
        return ""

    seen_docs = {}
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        doc_name = meta.get("document_name", "Unknown Document")
        if doc_name not in seen_docs:
            seen_docs[doc_name] = {
                "organization": meta.get("organization", "N/A"),
                "category": meta.get("category", "General"),
                "focus_area": meta.get("focus_area", "N/A"),
                "year": meta.get("year", "N/A"),
                "source_url": meta.get("source_url", ""),
                "pages": set(),
            }
        page = meta.get("page_number")
        if page and page != "N/A":
            seen_docs[doc_name]["pages"].add(str(page))

    citations = ["\n\n---\n##### Verified Sources & References"]
    for i, (doc_name, info) in enumerate(seen_docs.items(), 1):
        pages_str = f" | **Page(s)**: {', '.join(sorted(info['pages']))}" if info["pages"] else ""
        url_str = f" | [Source Link]({info['source_url']})" if info["source_url"] else ""
        citations.append(
            f"{i}. **{doc_name}** ({info['year']})\n"
            f"   - **Organization**: {info['organization']} | **Category**: {info['category']}\n"
            f"   - **Focus Area**: {info['focus_area']}{pages_str}{url_str}"
        )

    return "\n".join(citations)
