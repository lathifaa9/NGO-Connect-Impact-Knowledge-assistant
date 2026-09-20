"""
Text cleaner and normalizer for authentic documents.
Removes OCR noise, normalizes unicode, preserves markdown structure,
and extracts document metadata headers.
"""

import re
from typing import Dict, Tuple

def extract_header_metadata(text: str) -> Tuple[Dict[str, str], str]:
    """
    Parses key-value metadata fields embedded at the beginning of a markdown document.
    Example:
    **Document ID**: DOC-DARPAN-001
    **Category**: NGO Darpan / NITI Aayog
    Returns (metadata_dict, remaining_cleaned_text).
    """
    metadata: Dict[str, str] = {}
    lines = text.splitlines()
    remaining_lines = []
    parsing_header = True

    field_map = {
        "document id": "document_id",
        "category": "category",
        "organization": "organization",
        "publication year": "year",
        "year": "year",
        "source url": "source_url",
        "focus area": "focus_area",
        "operating area": "operating_area",
        "target beneficiaries": "target_beneficiaries",
    }

    for line in lines:
        stripped = line.strip()
        if parsing_header and (stripped.startswith("**") or stripped.startswith("#")):
            # Look for **Key**: Value
            match = re.match(r"\*\*(.+?)\*\*:\s*(.+)", stripped)
            if match:
                key = match.group(1).strip().lower()
                val = match.group(2).strip()
                if key in field_map:
                    metadata[field_map[key]] = val
                continue
            elif stripped.startswith("# "):
                # Title
                metadata["document_name"] = stripped.replace("# ", "").strip()
                continue
            elif stripped == "---":
                parsing_header = False
                continue
        elif parsing_header and stripped == "":
            continue
        else:
            parsing_header = False
            remaining_lines.append(line)

    cleaned_body = "\n".join(remaining_lines)
    return metadata, cleaned_body

def clean_text(text: str) -> str:
    """
    Normalizes text while preserving paragraph structure, bullet points, and headings.
    """
    if not text:
        return ""

    # Replace non-breaking spaces and odd unicode whitespace
    text = text.replace("\u00a0", " ").replace("\u200b", "")

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3 or more consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal whitespace on lines
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]

    return "\n".join(cleaned_lines).strip()
