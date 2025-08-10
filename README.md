# 📄 Multilingual LLM-Powered Intelligent Query–Retrieval System

## 📌 Problem Statement
Design and implement an **LLM-powered intelligent query–retrieval system** that can process large, unstructured, and multilingual documents to answer **contextual, domain-specific questions**.  
This solution is targeted for **insurance, legal, HR, and compliance** sectors, where accuracy, clarity, and explainability are critical.

---

## 🎯 Overview
We have comprehensively addressed the problem statement by creating a system that:

- **Ingests diverse document formats**: PDF, DOCX, email, scanned images, or even live JSON/HTML data from web/API sources.
- Performs **robust text extraction and preprocessing**, with OCR fallback for image-based documents.
- Uses **intelligent chunking** to preserve context.
- Generates **Azure OpenAI embeddings** and stores them in a **FAISS vector database** for fast semantic search.
- Allows **natural language questions** and returns **context-aware answers**, translated back to the question’s language if needed.
- Provides a **fully functional REST API** with live Swagger UI for easy testing and validation.

---

## 🚀 Features
✅ Multi-format ingestion (PDF, DOCX, email, JSON, HTML)  
✅ OCR fallback for scanned PDFs using `pytesseract`  
✅ Multilingual document and query support  
✅ Fast, context-aware retrieval using FAISS  
✅ Context-grounded answers with relevant clause quoting  
✅ Web scraping and API data extraction  
✅ Cached embeddings for improved performance on repeated queries  
✅ Built-in Swagger UI for testing

---

## 🛠️ Tech Stack
- **FastAPI** – REST API framework
- **LangChain** – LLM orchestration
- **Azure OpenAI** – LLM and embeddings
- **FAISS** – Vector database for semantic search
- **PyMuPDF**, **docx2txt**, **email.parser** – Document parsing
- **pytesseract** + **pdf2image** – OCR for scanned PDFs
- **BeautifulSoup** – HTML parsing

---

## 📂 Project Structure
