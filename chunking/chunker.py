"""
Semantic and recursive section-aware text chunker.
Splits documents into coherent chunks while preserving section boundaries,
page numbers, and injecting rich contextual metadata into each chunk.
"""

from typing import Any, Dict, List
from config.config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.document_loader import LoadedDocument
from ingestion.metadata import DocumentChunk
from utils.helpers import compute_hash
from utils.logger import logger

class TextChunker:
    """Chunks documents into overlapping segments with rich provenance metadata."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Splits text into paragraphs by double newlines."""
        raw_paras = text.split("\n\n")
        return [p.strip() for p in raw_paras if p.strip()]

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        """
        Chunks a loaded document into context-enriched chunks.
        """
        file_path = doc.file_path
        meta = doc.metadata.copy()

        # Derive folder-based category if not in header
        category_folder = file_path.parent.name
        cat_mapping = {
            "ngo_darpan": "NGO Darpan / NITI Aayog",
            "government_guidelines": "Government NGO Registration & Guidelines",
            "fcra": "FCRA Documents",
            "csr": "CSR Documents",
            "ngo_reports": "NGO Annual Reports",
            "annual_reports": "NGO Annual Reports",
            "project_reports": "NGO Program / Project Reports",
            "impact_reports": "NGO Impact Assessment Reports",
            "government_schemes": "Government Schemes Implemented Through NGOs",
            "un_reports": "UN / UN-related Reports",
            "established_ngo_reports": "Established NGO Reports",
        }
        category = meta.get("category", cat_mapping.get(category_folder, "General"))
        doc_name = meta.get("document_name", file_path.stem.replace("_", " ").title())
        org = meta.get("organization", "Non-Governmental / Statutory Body")
        focus = meta.get("focus_area", "Community Development")
        year = meta.get("year", "2023")
        page = doc.page_number
        source_url = meta.get("source_url", "")
        doc_id = meta.get("document_id", f"DOC-{compute_hash(file_path.name)[:8].upper()}")

        paragraphs = self._split_into_paragraphs(doc.content)
        chunks: List[DocumentChunk] = []

        current_text = ""
        chunk_index = 0

        for para in paragraphs:
            # If adding this paragraph exceeds chunk_size and we already have text
            if len(current_text) + len(para) > self.chunk_size and current_text:
                chunk_id = f"{doc_id}_{page}_{chunk_index}"
                context_prefix = (
                    f"[Document: {doc_name} | Org: {org} | Category: {category} | Focus: {focus} | Page: {page}]\n"
                )
                full_chunk_text = context_prefix + current_text.strip()

                chunk_meta: Dict[str, Any] = {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "category": category,
                    "organization": org,
                    "year": str(year),
                    "focus_area": focus,
                    "page_number": str(page),
                    "source_url": source_url,
                    "file_name": file_path.name,
                    "chunk_index": chunk_index,
                }

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=full_chunk_text,
                        metadata=chunk_meta,
                    )
                )
                chunk_index += 1

                # Keep overlap from the end of current_text
                overlap_text = current_text[-self.chunk_overlap :] if len(current_text) > self.chunk_overlap else ""
                current_text = overlap_text + "\n\n" + para
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para

        # Flush final chunk
        if current_text.strip():
            chunk_id = f"{doc_id}_{page}_{chunk_index}"
            context_prefix = (
                f"[Document: {doc_name} | Org: {org} | Category: {category} | Focus: {focus} | Page: {page}]\n"
            )
            full_chunk_text = context_prefix + current_text.strip()

            chunk_meta = {
                "document_id": doc_id,
                "document_name": doc_name,
                "category": category,
                "organization": org,
                "year": str(year),
                "focus_area": focus,
                "page_number": str(page),
                "source_url": source_url,
                "file_name": file_path.name,
                "chunk_index": chunk_index,
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=full_chunk_text,
                    metadata=chunk_meta,
                )
            )

        return chunks

    def chunk_all(self, loaded_docs: List[LoadedDocument]) -> List[DocumentChunk]:
        """Chunks a collection of loaded documents."""
        all_chunks = []
        for doc in loaded_docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(f"Generated {len(all_chunks)} chunks across {len(loaded_docs)} document sections.")
        return all_chunks
