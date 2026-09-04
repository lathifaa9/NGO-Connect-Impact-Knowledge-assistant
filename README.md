# NGO Connect & Impact Knowledge Assistant

## 📌 About the Project

**NGO Connect & Impact Knowledge Assistant** is an AI-powered chatbot designed to help users find reliable information about **Non-Governmental Organizations (NGOs)**, their activities, services, regulations, government schemes, and social impact.

The project uses **Retrieval-Augmented Generation (RAG)** to answer questions using information collected from trusted sources such as government documents, NGO reports, UN reports, FCRA and CSR documents, and other verified publications.

Instead of relying only on the knowledge stored inside an AI model, the system first searches its own knowledge base for relevant information and then uses that information to generate the answer.

---

## 🎯 Problem Statement

Information about NGOs is often spread across many different websites, government documents, reports, and lengthy PDF files. Finding the required information can be time-consuming, and users may also encounter unreliable or outdated information.

This project provides a single AI-based interface where users can ask questions in natural language and receive answers based on a curated collection of trusted NGO-related documents.

---

## 💡 What the System Can Answer

The assistant focuses on questions related to:

- What NGOs are and how they work
- NGO formation and registration
- NGO activities and areas of work
- NGO services and support
- How people can find and approach NGOs
- Government schemes implemented through NGOs
- FCRA and foreign contribution information
- CSR and NGO-related activities
- NGO annual and impact reports
- Contributions of NGOs to society
- Social and community impact

---

## 🧠 How It Works

The project follows a **Retrieval-Augmented Generation (RAG)** approach.

```text
User Question
      ↓
Question Processing
      ↓
Search Knowledge Base
      ↓
Retrieve Relevant Document Chunks
      ↓
Provide Context to LLM
      ↓
Generate Grounded Answer
      ↓
Answer + Sources
```

The documents are first processed and divided into smaller sections called **chunks**. These chunks are converted into embeddings and stored in a vector database.

When a user asks a question, the system searches for the most relevant chunks and provides them as context to the Large Language Model (LLM).

The LLM then generates an answer based on the retrieved information.

---

## 📚 Knowledge Base

The knowledge base contains authentic documents collected from trusted sources, including:

- Government organizations
- NGO Darpan / NITI Aayog
- FCRA resources
- CSR resources
- Government scheme documents
- UN organizations
- Established NGOs
- NGO annual reports
- NGO impact reports

The quality and reliability of the knowledge base are important because the assistant is designed to provide **source-grounded information**.

---

## 🔎 Source-Based Answers

The system provides references to the documents used to generate an answer.

Sources may include:

- Document name
- Organization
- Page number
- Document category
- Source URL

This allows users to verify the information instead of simply trusting the AI-generated response.

---

## 🛑 Handling Unknown Questions

The assistant is designed not to invent information.

If the required information cannot be found in the available knowledge base, the system can respond with a message such as:

> "I don't have enough information in the NGO knowledge base to answer that question."

This helps reduce unsupported answers and hallucinations.

---

## 🛠️ Technology Stack

- **Python** – Core programming language
- **Streamlit** – Frontend and user interface
- **LlamaIndex** – RAG framework
- **ChromaDB** – Vector database
- **Sentence Transformers** – Text embeddings
- **Llama-based LLM** – Answer generation
- **PyPDF** – PDF processing
- **pytest** – Testing
- **Git & GitHub** – Version control

---

## ✨ Key Features

- AI-powered NGO information assistant
- Retrieval-Augmented Generation (RAG)
- Trusted document-based knowledge base
- Semantic document search
- Source and page-level citations
- Knowledge-base-based answers
- Unknown/out-of-scope question handling
- Document library
- User-friendly chat interface
- RAG evaluation and testing

---

## 👥 Team

**DATA CODEX — TEAM 10**

1 Lathifaa

2 Akhil
3 Jayasri
4 Lohitha
5 Ramya

## 🚀 Project Goal

The goal of this project is to demonstrate how **Retrieval-Augmented Generation can be used to build a reliable and transparent AI knowledge assistant** that makes important NGO-related information easier to find, understand, and verify.

The project focuses on:

**Reliable Information • Source Transparency • RAG • AI • Usability**
