"""
Document Loader module for authentic NGO knowledge base documents.
Supports Markdown (.md), Plain Text (.txt), and PDF (.pdf) files with page-level tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os

from config.config import RAW_DOCS_DIR
from ingestion.text_cleaner import clean_text, extract_header_metadata
from utils.logger import logger

class LoadedDocument:
    """Represents a loaded source document or document page with metadata."""
    def __init__(
        self,
        file_path: Path,
        content: str,
        page_number: str = "1",
        metadata: Optional[Dict[str, str]] = None
    ):
        self.file_path = file_path
        self.content = content
        self.page_number = page_number
        self.metadata = metadata or {}

class DocumentLoader:
    """Recursively loads documents from raw_docs directory."""

    def __init__(self, raw_docs_dir: Path = RAW_DOCS_DIR):
        self.raw_docs_dir = Path(raw_docs_dir)

    def load_markdown_file(self, file_path: Path) -> List[LoadedDocument]:
        """Loads a markdown file, extracting top headers and page/section markers."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            extracted_meta, body_text = extract_header_metadata(raw_text)
            cleaned = clean_text(body_text)

            # Check if document has internal page markers (e.g. --- Page 2 ---)
            if "--- Page " in cleaned:
                pages = []
                segments = cleaned.split("--- Page ")
                # First segment is page 1
                if segments[0].strip():
                    pages.append(LoadedDocument(file_path, segments[0].strip(), "1", extracted_meta))
                for seg in segments[1:]:
                    lines = seg.split("\n", 1)
                    page_num = lines[0].replace("---", "").strip()
                    page_content = lines[1] if len(lines) > 1 else ""
                    if page_content.strip():
                        pages.append(LoadedDocument(file_path, page_content.strip(), page_num, extracted_meta))
                return pages
            else:
                return [LoadedDocument(file_path, cleaned, "1", extracted_meta)]
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
            return []

    def load_pdf_file(self, file_path: Path) -> List[LoadedDocument]:
        """Loads a PDF file page-by-page using PyMuPDF (fitz)."""
        loaded_pages = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                page_text = clean_text(page.get_text("text"))
                if page_text.strip():
                    loaded_pages.append(
                        LoadedDocument(
                            file_path=file_path,
                            content=page_text,
                            page_number=str(page_idx + 1),
                            metadata={"document_name": file_path.stem.replace("_", " ").title()}
                        )
                    )
            doc.close()
            logger.info(f"Loaded {len(loaded_pages)} pages from PDF: {file_path.name}")
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed; falling back to pypdf if available.")
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                for page_idx, page in enumerate(reader.pages):
                    text = clean_text(page.extract_text() or "")
                    if text.strip():
                        loaded_pages.append(
                            LoadedDocument(
                                file_path=file_path,
                                content=text,
                                page_number=str(page_idx + 1),
                                metadata={"document_name": file_path.stem.replace("_", " ").title()}
                            )
                        )
            except Exception as e:
                logger.error(f"Failed to load PDF {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")

        return loaded_pages

    def load_all_documents(self) -> List[LoadedDocument]:
        """Scans the raw_docs directory and loads all supported documents."""
        all_docs: List[LoadedDocument] = []
        if not self.raw_docs_dir.exists():
            logger.warning(f"Raw docs directory {self.raw_docs_dir} does not exist.")
            return all_docs

        for root, _, files in os.walk(self.raw_docs_dir):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext in [".md", ".markdown", ".txt"]:
                    docs = self.load_markdown_file(file_path)
                    all_docs.extend(docs)
                elif ext == ".pdf":
                    docs = self.load_pdf_file(file_path)
                    all_docs.extend(docs)

        logger.info(f"Total document pages/sections loaded: {len(all_docs)}")
        return all_docs
