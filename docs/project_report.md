# B.Tech Final Year Capstone Project Report
## NGO Connect & Impact Knowledge Assistant
### A Source-Grounded Retrieval-Augmented Generation (RAG) Platform for Indian Civil Society & Statutory Frameworks

**Department of Computer Science & Engineering**  
**Project Group:** DATA CODEX — TEAM 10  
**Authors & Contributors:**
1. Lathifaa
2. Akhil
3. Jayasree
4. Lohitha
5. Ramya

---

## 1. Abstract

Non-Governmental Organizations (NGOs) and Civil Society Organizations (CSOs) play a transformative role in driving socio-economic progress, grassroots welfare, environmental conservation, and disaster resilience across India. However, the regulatory, statutory, and programmatic information governing the sector—spanning NITI Aayog NGO Darpan registration, Foreign Contribution (Regulation) Act (FCRA 2020), Corporate Social Responsibility (CSR) Section 135 rules, and audited impact assessments—remains fragmented across heterogeneous ministry portals, dense statutory gazettes, and lengthy annual reports. Consequently, citizens, social sector leaders, donors, and researchers struggle to obtain verifiable, transparent answers. Standard Large Language Models (LLMs) frequently hallucinate statutory details and lack verifiable citations.

This capstone project presents **NGO Connect & Impact Knowledge Assistant**, an end-to-end Retrieval-Augmented Generation (RAG) system engineered to deliver source-grounded, verifiable answers to natural language inquiries. Built on a curated corpus of 21 authentic policy frameworks, statutory guidelines, and audited NGO reports, the architecture integrates dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`), persistent vector indexing via ChromaDB, strict cosine-distance guardrail thresholding (maximum cosine distance 0.85 / minimum similarity 0.25), and multi-engine grounded synthesis (Llama-3.1-8B via Groq API and an offline Grounded Extractive Synthesizer). Evaluated across a comprehensive 15-query benchmark suite, the platform achieves a **100.0% Hit Rate @ K=5**, an **MRR of 0.836**, and **100.0% Out-of-Scope Guardrail Rejection Accuracy** with an average pipeline latency of **24.1 ms**.

---

## 2. Introduction & Problem Statement

### 2.1 Background
India possesses one of the world's largest voluntary sectors, comprising over 3 million registered non-profit entities operating under varied legal formats:
- **Public Charitable Trusts** (governed by State Trust Acts or the Indian Trusts Act, 1882)
- **Societies** (registered under the Societies Registration Act, 1860)
- **Section 8 Companies** (incorporated under the Companies Act, 2013)

To mobilize public donations, international grants, or corporate CSR investments, NGOs must comply with an intricate web of statutory mandates:
- **NITI Aayog NGO Darpan Portal**: Mandatory Unique Identification for receiving central ministry grants.
- **FCRA (Foreign Contribution Regulation Act, 2020 Amendment)**: Mandatory SBI New Delhi Main Branch designated accounts, cap on administrative expenses at 20%, and prohibition on sub-granting foreign funds.
- **Section 135 of Companies Act, 2013 & CSR Rules**: Minimum 2% net profit allocation for qualifying companies, mandatory registration of implementing agencies via Form CSR-1, and 5% administrative overhead ceilings.
- **Income Tax Exemptions**: Continuous validation under Sections 12A/12AB and 80G.

### 2.2 Problem Formulation
1. **Information Asymmetry & Dispersion**: Regulatory guidelines and impact filings are scattered across disparate portals (mha.gov.in, mca.gov.in, ngodarpan.gov.in, aajeevika.gov.in) and hundred-page PDF annual reports.
2. **Hallucination & Legal Liability in Vanilla LLMs**: Public LLMs (e.g., ChatGPT, Gemini) exhibit severe hallucination risks when asked about compliance deadlines, Section 8 thresholds, or FCRA requirements, inventing non-existent rules without document traceability.
3. **Absence of Grounded Citations**: General search engines fail to map specific statutory clauses or audited outcome figures to verified source documents.

---

## 3. Objectives & Technical Contributions

The core objectives of the project are:
1. **Curate an Authentic Statutory & Impact Corpus**: Assemble a multi-thematic corpus of 21 authentic documents covering NITI Aayog policies, FCRA amendments, MCA CSR rules, central government schemes (DAY-NRLM, DDRS), and audited annual reports from leading NGOs (Pratham, Goonj, Akshaya Patra, HelpAge India, SEWA, CSE).
2. **Design an Overlapping Semantic Chunking Pipeline**: Segment statutory documents into 512–650 token chunks with 64–120 token overlaps to preserve semantic continuity across legal sub-clauses.
3. **Deploy Dense Vector Indexing in ChromaDB**: Map chunks into 384-dimensional dense vector space using `sentence-transformers/all-MiniLM-L6-v2` and persist embeddings in ChromaDB with cosine distance indexing.
4. **Implement Strict Anti-Hallucination Guardrails**: Enforce a minimum cosine similarity threshold of 0.25 (maximum cosine distance 0.85). If the retrieved chunks fail relevance scoring or fall out-of-scope, the pipeline strictly returns:
   > *"I don't have enough information in the NGO knowledge base to answer that question."*
5. **Multi-Model Synthesis & Intent Routing**: Build a conversational intent router for pleasantries/capabilities questions, paired with dual synthesis engines: high-speed Groq Llama-3.1-8B and a zero-dependency offline Grounded Extractive Synthesizer.
6. **Production Streamlit Dashboard**: Develop an intuitive, multi-session user interface featuring chat history, key takeaway extraction, verified source cards with direct URLs, a filterable NGO Directory, a Document Library, automated benchmark evaluations, and system telemetry.

---

## 4. Literature Review & Comparative Analysis

| Feature / Metric | Vanilla Pre-Trained LLMs | Fine-Tuned Domain LLMs | Retrieval-Augmented Generation (Our Approach) |
| :--- | :--- | :--- | :--- |
| **Knowledge Recency** | Static cut-off date | Outdated post fine-tuning | Dynamically updatable via vector store |
| **Hallucination Risk** | High (generative speculation) | Moderate (unpredictable outputs) | Minimal / Zero (strictly grounded context) |
| **Source Traceability** | None (no source links) | None (weights store knowledge) | 100% Granular (document, organization, page, URL) |
| **Computational Cost** | High API token cost | Extremely high GPU training cost | Low (one-time local embedding + fast retrieval) |
| **Statutory Guardrails** | Weak / Unenforced | Hard to constrain deterministically | Deterministic thresholding (< 0.25 similarity rejects) |

RAG combines the retrieval power of dense semantic search with the linguistic coherence of modern language models, ensuring that every answer is backed by verified statutory documentation.

---

## 5. System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE (Streamlit)                    │
│  [Chat Assistant]  [NGO Directory]  [Doc Library]  [Benchmarks]  [Diag]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Natural Language Query
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CONVERSATIONAL INTENT ROUTER                    │
│   • Greetings / Introductions  ──► Warm Contextual Capability Intro    │
│   • Domain Inquiries           ──► Proceed to Semantic Retrieval       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     HYBRID RETRIEVAL & VECTOR ENGINE                   │
│   • Query Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-d)    │
│   • Persistent Storage: ChromaDB ('ngo_knowledge_base', 155 Chunks)    │
│   • Metadata Filtering: Category, Focus Area, Organization             │
│   • Metric: Cosine Similarity = 1.0 - Cosine Distance                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Top-K Retrieved Chunks (K=5)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       RESPONSE VALIDATOR & GUARDRAIL                   │
│   • If max(similarity) < 0.25 OR Out-of-Scope Query:                   │
│     ──► Fallback: "I don't have enough information in the NGO..."      │
│   • If Valid: Context Assembly & Prompt Construction                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Grounded Context
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     SYNTHESIS & ANSWER GENERATION                      │
│   • Engine A: Groq Cloud API (Llama-3.1-8B-Instant)                    │
│   • Engine B: Grounded Extractive Synthesizer (Offline Zero-API)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Structured Answer
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  CITATIONS & PRESENTATION FORMATTING                   │
│   • Key Points Extraction (Top 3-4 actionable bullets)                 │
│   • Verified Source Cards (Document Name, Organization, Year, URL)     │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Corpus Ingestion & Metadata Taxonomy
The corpus encompasses 21 primary documents organized across 7 functional categories:
1. **NGO Darpan / NITI Aayog**: Guidelines for Voluntary Sector Policy, Registration Manual.
2. **FCRA Compliance**: Foreign Contribution Regulation Act 2020 Rules, SBI Account Guidelines.
3. **Corporate Social Responsibility (CSR)**: MCA Section 135 Rules, Form CSR-1 Registration FAQs.
4. **Government Schemes**: DAY-NRLM Partnership Guidelines, MSJE DDRS Disability Grants.
5. **Education Sector Impact**: Pratham Annual Report & ASER Foundational Learning Assessments.
6. **Livelihoods & Rural Welfare**: Goonj Disaster Relief & Cloth for Work, SEWA Women's Reports.
7. **Healthcare & Sustainability**: HelpAge India Healthcare Report, Sightsavers India, CSE Environment.

Each document is tagged with structured metadata: `document_id`, `document_name`, `organization`, `category`, `focus_area`, `year`, `source_url`, and `target_beneficiaries`.

### 5.2 Mathematical Formulations

#### 5.2.1 Cosine Similarity
Given query embedding vector $\mathbf{q} \in \mathbb{R}^{384}$ and document chunk embedding vector $\mathbf{d}_i \in \mathbb{R}^{384}$:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\|_2 \|\mathbf{d}_i\|_2}$$

ChromaDB computes cosine distance:
$$\text{Cosine Distance}(\mathbf{q}, \mathbf{d}_i) = 1.0 - \text{Cosine Similarity}(\mathbf{q}, \mathbf{d}_i)$$

#### 5.2.2 Guardrail Decision Function
Let $\mathcal{D}_K = \{\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_K\}$ be the top-$K$ retrieved chunks. The grounding decision $G(q)$ is defined as:
$$G(q) = \begin{cases} \text{Synthesize Answer}, & \text{if } \max_{i \in [1, K]} \text{Similarity}(\mathbf{q}, \mathbf{d}_i) \ge 0.25 \land \text{Overlap}(\mathbf{q}, \mathcal{D}_K) > 0 \\ \text{Trigger Fallback}, & \text{otherwise} \end{cases}$$

#### 5.2.3 Mean Reciprocal Rank (MRR)
For $N$ benchmark queries, where $\text{rank}_i$ denotes the rank position of the first relevant document:
$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

---

## 6. Experimental Results & Benchmark Evaluation

The system was evaluated against a 15-question benchmark suite comprising 12 supported domain inquiries and 3 out-of-scope queries designed to test anti-hallucination guardrails.

### 6.1 Quantitative Summary

| Metric | Measured Score | Evaluation Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Supported Queries Hit Rate @ K=5** | **100.0%** (12 / 12) | $\ge 90.0\%$ | **EXCEEDED** |
| **Mean Reciprocal Rank (MRR)** | **0.836** | $\ge 0.750$ | **EXCEEDED** |
| **Guardrail Rejection Accuracy** | **100.0%** (3 / 3) | $100.0\%$ | **PERFECT** |
| **Answer Grounding Pass Rate** | **100.0%** (12 / 12) | $\ge 90.0\%$ | **EXCEEDED** |
| **Average End-to-End Latency** | **24.1 ms** (CPU Inference) | $\le 250\text{ ms}$ | **OPTIMAL** |

### 6.2 Detailed 15-Question Benchmark Matrix

| ID | Benchmark Query | Category / Focus Area | Domain Type | Best Cosine Similarity | Retrieval Result | Grounding Verification |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Q01** | What is an NGO? | Public Policy / Governance | Supported | 0.672 | PASS | GROUNDED (100%) |
| **Q02** | How are NGOs registered in India? | Statutory Registration | Supported | 0.741 | PASS | GROUNDED (100%) |
| **Q03** | How can a person find an NGO in an area? | NITI Aayog NGO Darpan | Supported | 0.607 | PASS | GROUNDED (100%) |
| **Q04** | What services and support do NGOs provide? | Grassroots Development | Supported | 0.661 | PASS | GROUNDED (100%) |
| **Q05** | Which NGOs work in education & literacy? | Primary Education | Supported | 0.604 | PASS | GROUNDED (100%) |
| **Q06** | Which NGOs work in healthcare & senior care? | Elderly Welfare / Healthcare | Supported | 0.595 | PASS | GROUNDED (100%) |
| **Q07** | Which NGOs work in women empowerment? | Livelihoods / Self-Help Groups | Supported | 0.658 | PASS | GROUNDED (100%) |
| **Q08** | Which NGOs work in environment & clean air? | Environmental Sustainability | Supported | 0.573 | PASS | GROUNDED (100%) |
| **Q09** | What is FCRA and its compliance rules? | Foreign Contribution Act | Supported | 0.638 | PASS | GROUNDED (100%) |
| **Q10** | What are the relevant CSR funding guidelines? | Corporate Social Responsibility | Supported | 0.710 | PASS | GROUNDED (100%) |
| **Q11** | What government schemes are for NGOs? | Central Assistance Schemes | Supported | 0.628 | PASS | GROUNDED (100%) |
| **Q12** | What impact was reported by Goonj? | Disaster Relief / Livelihood | Supported | 0.595 | PASS | GROUNDED (100%) |
| **Q13** | Who won the football World Cup in 1998? | Unrelated Sports Domain | Out-of-Scope | 0.000 | PASS (Filtered) | REJECTED (Zero Hallucination) |
| **Q14** | What is the stock price of Tesla Motors? | Financial Market Query | Out-of-Scope | 0.226 | PASS (Filtered) | REJECTED (Zero Hallucination) |
| **Q15** | What is the capital and currency of Mars? | Fictional / Astronomical Query | Out-of-Scope | 0.000 | PASS (Filtered) | REJECTED (Zero Hallucination) |

---

## 7. Streamlit User Interface & System Modules

The frontend is implemented in Streamlit with modern custom CSS styling, offering 5 distinct views accessible via the sidebar:
1. **NGO Assistant / Home**:
   - Conversational chat stream with multi-session history (new chat, rename, delete, clear all).
   - Conversational routing for instant greetings and capabilities guidance.
   - User queries aligned on the right in purple gradient bubbles; Assistant responses aligned on the left in white bordered cards.
   - Extracted key takeaways box with accent borders.
   - Interactive source citation cards featuring document name, publishing organization, year, and direct reference links.
   - Attachment popover for contextual document inspection.
2. **NGO Directory**:
   - Searchable registry of verified organizations with filters for thematic focus areas.
   - Displays DARPAN IDs, FCRA registration status, CSR-1 status, audited outcome metrics, and central scheme partnerships.
3. **Document Corpus Library**:
   - Filterable catalog of all 21 raw authentic policy and reporting documents.
   - Metadata breakdown and direct links to official government and non-profit portals.
4. **Evaluation & Benchmarks**:
   - Real-time KPI display for Hit Rate @ 5, MRR, Guardrail Accuracy, and Latency.
   - Complete 15-question benchmark matrix table.
   - One-click triggers for live retrieval and grounding benchmark test runs.
5. **System Diagnostics & Architecture**:
   - Vector store telemetry, indexed chunk counts, and embedding model specifications.
   - In-app "Re-Index Knowledge Base" control executing the automated ingestion pipeline.

---

## 8. Ethical Considerations & Limitations

1. **Information Accuracy & Grounding**: The system explicitly decouples itself from open-ended generative extrapolation. If a document does not contain an answer, it admits lack of information rather than misleading a user on statutory matters.
2. **Fair Sectoral Representation**: The corpus reflects balanced representation across education, healthcare, livelihood, elderly care, environment, and disability sectors.
3. **Limitations**:
   - The knowledge base covers curated central government statutes and selected NGO reports; state-specific non-profit rules vary across regional jurisdictions.
   - Extractive synthesis relies on textual presence; highly abstract cross-document syntheses are enhanced when an external Groq API key is supplied.

---

## 9. Conclusion & Future Roadmap

The **NGO Connect & Impact Knowledge Assistant** successfully illustrates how modern Retrieval-Augmented Generation can transform complex statutory and social impact data into an accessible, transparent, and hallucination-free decision-support platform. By combining dense semantic embeddings, persistent vector indexing, strict similarity guardrails, and an intuitive Streamlit interface, the project fulfills all objectives of the B.Tech CSE Final Year Capstone Project.

### Future Enhancements:
- **Multilingual Support**: Incorporate IndicBERT and multilingual Sentence Transformers to enable inquiries in Hindi, Telugu, Tamil, and other regional Indian languages.
- **Automated Web-Scraping Pipeline**: Periodically refresh MCA and NITI Aayog circulars via scheduled ingestion sidecars.
- **Voice-Enabled Interface**: Integrate speech-to-text (Whisper) and text-to-speech for rural social workers and community volunteers.

---

## 10. References & Statutory Citations

1. Ministry of Home Affairs, Government of India. *The Foreign Contribution (Regulation) Act, 2010 and Foreign Contribution (Regulation) Amendment Act, 2020*.
2. Ministry of Corporate Affairs, Government of India. *Companies (Corporate Social Responsibility Policy) Rules, 2014 & Section 135 of the Companies Act, 2013*.
3. NITI Aayog, Government of India. *National Policy on the Voluntary Sector & Guidelines for Registration on NGO Darpan Portal*, 2023.
4. Ministry of Rural Development, Government of India. *Deendayal Antyodaya Yojana - National Rural Livelihoods Mission (DAY-NRLM) Partnership Guidelines with Civil Society Organizations*, 2023.
5. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS).
6. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
7. Chroma Core Team. *Chroma: The AI-native open-source embedding database*, 2024.
8. Pratham Education Foundation. *Annual Report & Annual Status of Education Report (ASER)*, 2023–24.
9. Goonj. *Annual Impact Report: Cloth for Work & Disaster Relief Initiatives*, 2023.
