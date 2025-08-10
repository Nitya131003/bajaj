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

## How it works (step-by-step)

1. Client POSTs to `/api/v1/hackrx/run` with `documents` (comma-separated URLs) and `questions` (list of queries).

2. For each URL the server attempts to fetch raw content. If the URL returns HTML/JSON/plain token text it may be used directly (useful for simple API endpoints).

3. Otherwise the URL is downloaded and file type is detected; extraction is done with:
   - **PDFs:** PyMuPDF text extraction; fallback: convert pages to images (`pdf2image`) + OCR (`pytesseract`).
   - **DOCX:** `docx2txt`
   - **EML:** `email.parser.BytesParser`

4. Extracted text is normalized (remove hyphen-newlines, collapse whitespace).

5. If the document language is not English, the LLM is used to translate the document into English for indexing.

6. Text is chunked (`chunk_size=1200`, `chunk_overlap=200`) and embedded using Azure embeddings.

7. A `RetrievalQA` chain is created using the FAISS retriever and a conservative prompt template (`GENERIC_PROMPT`) that instructs the LLM to answer only from context and avoid hallucination.

8. For each question the system either runs a specialized procedural flow or routes the question to the chain. The final answer is returned in an `answers` array.

---

## Caching & Persistence
- Uses **FAISS** for persistent local storage.  
- Cache prevents redundant embedding generation.  

## Prompting & Explainability
- Prompts are crafted to extract relevant, context-aware answers.  
- Debug logs explain processing steps.    

## Troubleshooting & Known Issues
- Large documents may need chunking.  
- Network timeouts if document URLs are slow.  
- Check permissions for persistent storage directory.  

## Example: Sample Policy Test (Knee Surgery)
- Upload medical policy documents.  
- **Query**: "Is knee surgery covered under policy X?"  
- System retrieves relevant section and returns an answer.

---

## Contribution
**Team Members:**
- Yogya Asnani  
- Chetan Sharma  
- Arya Singh  
- Nitya Jain  
- Radhika Jain

