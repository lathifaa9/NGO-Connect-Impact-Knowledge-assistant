"""
Configuration module for NGO Connect & Impact Knowledge Assistant.
Defines system paths, embedding settings, vector database parameters,
LLM configuration, and grounding constraints.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

# Prioritize data/raw_docs if present, fallback to data/raw
RAW_DOCS_DIR = DATA_DIR / "raw_docs" if (DATA_DIR / "raw_docs").exists() else DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCS_DIR = BASE_DIR / "docs"
EVAL_DIR = BASE_DIR / "evaluation"

# Create directories if they do not exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Processed Catalogs & Benchmarks
DOCUMENTS_CATALOG_PATH = PROCESSED_DIR / "documents_catalog.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
NGO_DIRECTORY_PATH = PROCESSED_DIR / "ngo_directory.json"
TEST_QUESTIONS_PATH = EVAL_DIR / "test_questions.json"

# Text Chunking Settings
CHUNK_SIZE = 650
CHUNK_OVERLAP = 120

# Embedding Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "ngo_knowledge_base"
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
# Cosine distance threshold for grounding (distance <= 0.75 means relevant)
MAX_COSINE_DISTANCE = 0.85

# LLM Providers: "openai", "ollama", "extractive"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Strict Anti-Hallucination Fallback Message (Mandatory Rule #10)
UNSUPPORTED_ANSWER_MESSAGE = (
    "I don't have enough information in the NGO knowledge base to answer that question."
)

# Standard Focus Areas
FOCUS_AREAS = [
    "All Focus Areas",
    "Education",
    "Healthcare",
    "Women empowerment",
    "Child welfare",
    "Environment",
    "Rural development",
    "Livelihood / skill development",
    "Community development",
]

# Standard Categories
CATEGORIES = [
    "All Categories",
    "NGO Darpan / NITI Aayog",
    "Government NGO Registration & Guidelines",
    "FCRA Documents",
    "CSR Documents",
    "NGO Annual Reports",
    "NGO Program / Project Reports",
    "NGO Impact Assessment Reports",
    "Government Schemes Implemented Through NGOs",
    "UN / UN-related Reports",
    "Established NGO Reports",
]
