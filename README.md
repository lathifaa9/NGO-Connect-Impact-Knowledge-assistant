# NGO Connect & Impact Knowledge Assistant

> **A Retrieval-Augmented Generation (RAG) based AI assistant for reliable, source-grounded information about NGOs, their work, regulations, services, and social impact.**

---

## 📌 Project Overview

The **NGO Connect & Impact Knowledge Assistant** is an AI-powered question-answering system that helps users understand Non-Governmental Organizations (NGOs), their activities, regulations, services, and impact on society.

The system uses **Retrieval-Augmented Generation (RAG)** to generate answers based on a curated knowledge base of authentic documents from government organizations, NGOs, UN organizations, and other trusted sources.

Unlike a general-purpose chatbot, the system is designed to answer questions **only using information available in its verified knowledge base**.

When sufficient information is not available, the system clearly informs the user instead of generating unsupported information.

---

## 🎯 Problem Statement

Information about NGOs is distributed across government websites, regulations, annual reports, impact reports, scheme documents, and other official publications.

Users often face difficulties in:

- Finding trustworthy NGO information
- Understanding NGO registration and regulations
- Finding information about NGO services
- Understanding NGO contributions to society
- Identifying relevant government schemes
- Locating information inside lengthy reports
- Distinguishing reliable information from unsupported online content

This project provides a single AI-powered interface for searching and understanding this information.

---

## 💡 Objectives

The main objectives of this project are:

1. Build a reliable NGO-focused knowledge base.
2. Process authentic PDF documents from trusted sources.
3. Use RAG to retrieve relevant information before generating an answer.
4. Provide answers grounded in the retrieved documents.
5. Display document and page-level citations.
6. Prevent unsupported or hallucinated answers.
7. Clearly handle questions outside the knowledge base.
8. Provide a simple and professional user interface.
9. Evaluate retrieval and answer quality systematically.

---

# 🚀 Key Features

### 🔎 RAG-Based Question Answering

Users can ask natural-language questions about NGOs.

Example:

> What is the role of NGOs in implementing government schemes?

The system retrieves relevant information from the knowledge base and generates a grounded response.

---

### 📚 Source-Grounded Answers

Every answer is based on retrieved documents rather than relying only on the LLM's internal knowledge.

---

### 🔗 Document Citations

Answers provide source information such as:

- Document title
- Organization
- Page number
- Document category
- Source URL

Example:

```text
Source: FCRA Guidelines
Page: 27
Organization: Government of India
```

---

### 🛑 "I Don't Know" Handling

If the required information is not available in the knowledge base, the system does not attempt to invent an answer.

Example:

```text
I don't have enough information in the NGO
knowledge base to answer that question.
```

---

### 📖 Document Library

Users can explore the documents available in the knowledge base.

Documents can be organized by categories such as:

- Government
- FCRA
- CSR
- NGO Reports
- Impact Reports
- UN Reports
- Government Schemes

---

### 🧪 Evaluation System

The project includes an evaluation framework for measuring:

- Retrieval quality
- Answer correctness
- Citation accuracy
- Groundedness
- Refusal accuracy

---

# 🧠 How RAG Works

The system follows a Retrieval-Augmented Generation architecture.

```text
                    USER QUESTION
                          │
                          ▼
                 ┌─────────────────┐
                 │    FRONTEND     │
                 │    Streamlit    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   RAG SERVICE   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    RETRIEVER    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    ChromaDB     │
                 │ Vector Database  │
                 └────────┬────────┘
                          │
                          ▼
                  Relevant Chunks
                          │
                          ▼
                 ┌─────────────────┐
                 │       LLM       │
                 │   Generation    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Answer + Sources│
                 └────────┬────────┘
                          │
                          ▼
                       USER
```

---

# 🔄 RAG Pipeline

## 1. Document Collection

Authentic documents are collected from trusted sources such as:

- Government organizations
- NGO Darpan
- NITI Aayog
- FCRA resources
- CSR resources
- UN organizations
- Established NGOs
- Government scheme documentation
- NGO annual reports
- NGO impact reports

---

## 2. PDF Text Extraction

PDF documents are processed and their text is extracted.

The system works best with:

- Searchable PDFs
- Text-based PDFs
- Properly structured documents
- Clear page boundaries

Scanned documents may require OCR.

---

## 3. Text Cleaning

Extracted text is cleaned to remove unnecessary elements such as:

- Repeated headers
- Repeated footers
- Excessive whitespace
- Broken formatting
- Navigation artifacts

Important information is preserved.

---

## 4. Chunking

Large documents are divided into smaller pieces called **chunks**.

Initial configuration:

```text
Chunk Size: 512 tokens
Chunk Overlap: 64 tokens
```

Chunking makes it possible for the retrieval system to find specific sections of a large document.

---

## 5. Embeddings

Each chunk is converted into a numerical representation called an **embedding**.

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings.

These embeddings allow semantically similar questions and document sections to be matched.

---

## 6. Vector Database

The embeddings and their metadata are stored in:

```text
ChromaDB
```

Stored information includes:

```text
Document
Chunk
Embedding
Page Number
Organization
Category
Year
Source URL
```

---

## 7. Retrieval

When the user asks a question, the question is converted into an embedding.

The system searches ChromaDB for the most relevant chunks.

Initial retrieval configuration:

```text
Top-K = 3
```

The retrieved chunks become the evidence used by the LLM.

---

## 8. LLM Generation

The retrieved information is provided to the LLM together with a strict grounding prompt.

The LLM is instructed to:

- Use only retrieved context
- Avoid unsupported claims
- Avoid inventing information
- Clearly state when information is unavailable
- Provide source references

---

## 9. Final Answer

The user receives:

```text
Answer
+
Source Documents
+
Page Numbers
+
Relevant Metadata
```

This makes the response more transparent and verifiable.

---

# 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| RAG Framework | LlamaIndex |
| LLM | Llama-based model via Groq/Ollama |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Testing | pytest |
| Version Control | Git & GitHub |

---

# 📁 Project Structure

```text
ngo-connect-impact-assistant/
│
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── .claude/
│   ├── settings.json
│   └── commands/
│       ├── build-index.md
│       ├── run-app.md
│       ├── test.md
│       └── evaluate.md
│
├── data/
│   ├── raw/
│   │   ├── niti_darpan/
│   │   ├── registration_guidelines/
│   │   ├── fcra/
│   │   ├── csr/
│   │   ├── annual_reports/
│   │   ├── impact_reports/
│   │   ├── government_schemes/
│   │   ├── un_reports/
│   │   └── ngo_reports/
│   │
│   ├── processed/
│   └── document_registry.csv
│
├── chroma_db/
│
├── src/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── metadata.py
│   │
│   ├── retrieval/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── answer_generator.py
│   │
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── citation_service.py
│   │   └── evaluation_service.py
│   │
│   └── utils/
│       ├── logging_utils.py
│       └── validators.py
│
├── app/
│   ├── app.py
│   │
│   ├── components/
│   │   ├── chat_interface.py
│   │   ├── source_cards.py
│   │   ├── sidebar_filters.py
│   │   └── feedback_widget.py
│   │
│   └── pages/
│       ├── 1_About.py
│       ├── 2_Document_Library.py
│       └── 3_Evaluation_Dashboard.py
│
├── scripts/
│   ├── build_index.py
│   ├── inspect_chunks.py
│   ├── run_evaluation.py
│   └── reset_database.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation_questions.csv
│
├── docs/
│   ├── architecture.md
│   ├── data_sources.md
│   ├── testing_strategy.md
│   ├── team_roles.md
│   └── report_outline.md
│
└── outputs/
    ├── screenshots/
    ├── demo_video/
    └── final_report/
```

---

# 📊 RAG Configuration

The initial RAG configuration is:

```text
Documents              : 25–40 target
Chunk Size             : 512 tokens
Chunk Overlap          : 64 tokens
Embedding Model        : all-MiniLM-L6-v2
Embedding Dimension    : 384
Initial Top-K          : 3
Vector Database        : ChromaDB
```

These values are starting points and can be evaluated and optimized based on retrieval performance.

---

# 🔐 Reliability and Hallucination Control

The system uses multiple mechanisms to improve reliability.

### 1. Trusted Sources

The knowledge base prioritizes authentic government, UN, and established NGO documents.

### 2. Retrieval Before Generation

The LLM receives relevant retrieved evidence instead of being asked to answer without context.

### 3. Strict Prompting

The LLM is instructed not to introduce unsupported information.

### 4. Retrieval Threshold

Low-quality retrieval results can be rejected instead of blindly passed to the LLM.

### 5. Source Citations

Answers include document and page references.

### 6. Out-of-Knowledge-Base Refusal

Questions that cannot be answered using available documents should receive a clear refusal.

---

# 🧪 Testing Strategy

The evaluation dataset contains different categories of questions.

### Direct Questions

Questions whose answers should exist clearly in one document.

### Multi-Document Questions

Questions requiring information from multiple sources.

### Discovery Questions

Questions requiring the system to find relevant information across the knowledge base.

### Ambiguous Questions

Questions where the system needs to handle insufficient context.

### Out-of-Knowledge-Base Questions

Questions unrelated to the available NGO knowledge base.

---

# 📈 Evaluation Metrics

The system can be evaluated using:

| Metric | Purpose |
|---|---|
| Retrieval Precision | Measures relevance of retrieved chunks |
| Retrieval Recall | Measures whether useful evidence was retrieved |
| Answer Correctness | Measures factual correctness |
| Citation Accuracy | Measures whether citations support answers |
| Groundedness | Measures whether answers stay within retrieved context |
| Refusal Accuracy | Measures handling of unavailable information |

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd ngo-connect-impact-assistant
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```text
GROQ_API_KEY=your_api_key_here
```

Never commit API keys or secrets to GitHub.

---

# 🗂️ Build the Knowledge Base

Place verified PDF documents inside the appropriate directories under:

```text
data/raw/
```

Then run the indexing process:

```bash
python scripts/build_index.py
```

This processes the documents, creates chunks, generates embeddings, and stores them in ChromaDB.

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app/app.py
```

The application will open in the browser.

---

# 🧪 Run Tests

Run the test suite using:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

# 📊 Run Evaluation

The evaluation system can be executed using:

```bash
python scripts/run_evaluation.py
```

The evaluation questions are stored in:

```text
tests/evaluation_questions.csv
```

---

# 👥 Team Roles

## Member 1 — Team Lead + Backend/Integration

Responsibilities:

- System architecture
- Backend integration
- RAG orchestration
- Git/GitHub management
- Configuration
- Final integration

---

## Member 2 — Data & Document Engineer

Responsibilities:

- PDF collection
- Source verification
- PDF processing
- Text cleaning
- Metadata
- Knowledge-base management

---

## Member 3 — RAG / AI Engineer

Responsibilities:

- Chunking
- Embeddings
- ChromaDB
- Retrieval
- Similarity search
- Reranking
- Retrieval optimization

---

## Member 4 — Frontend + LLM Engineer

Responsibilities:

- Streamlit interface
- Chat interface
- UI components
- LLM integration
- Prompt engineering
- Answer generation
- Citation presentation

---

## Member 5 — QA + Evaluation + DevOps

Responsibilities:

- Unit testing
- Integration testing
- RAG evaluation
- Performance testing
- Deployment
- Documentation
- Final quality assurance

---

# 🔮 Future Enhancements

Possible future improvements include:

- Hybrid semantic + keyword search
- Advanced reranking
- Conversational memory
- Multilingual NGO information
- Better OCR support
- Document upload functionality
- Advanced document filtering
- Query rewriting
- Improved evaluation dashboard
- Feedback-driven retrieval improvement
- Multimodal document understanding
- Cloud deployment
- Authentication and user management

---

# 🎓 Learning Outcomes

Through this project, the team gains practical experience in:

- Retrieval-Augmented Generation
- Large Language Models
- Embeddings
- Vector databases
- Semantic search
- Prompt engineering
- PDF processing
- Backend development
- Frontend development
- Software testing
- AI evaluation
- Git/GitHub
- Deployment
- Technical documentation

---

# 📜 Project Scope

The system focuses on information available in its curated knowledge base.

It is intended to assist users in discovering and understanding NGO-related information and **should not be treated as a substitute for official legal, financial, regulatory, or professional advice**.

For regulatory or legal matters, users should verify the information against the latest official source documents.

---

# ⭐ Project Vision

The goal of **NGO Connect & Impact Knowledge Assistant** is to demonstrate how Retrieval-Augmented Generation can transform large collections of trustworthy NGO and public-sector documents into an accessible, transparent, and source-grounded knowledge assistant.

The project prioritizes:

```text
Reliability
    +
Transparency
    +
Retrieval Quality
    +
Useful User Experience
    +
Responsible AI
```

---

## 👨‍💻 Team

**DATA CODEX — TEAM 10**
1 Lathifaa
2 Akhil
3 Jayasri
4 Lohitha
5 Ramya

**Project:** NGO Connect & Impact Knowledge Assistant

Built as a collaborative RAG/AI project focused on making reliable NGO information easier to discover and understand.
