import os
import requests
import asyncio
import hashlib
import json
import re
import gc
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import docx2txt
from email import policy
from email.parser import BytesParser
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from fastapi.responses import Response

# Optional OCR dependencies
from pdf2image import convert_from_bytes
import pytesseract

load_dotenv()

app = FastAPI(
    title="AI Insurance Analyzer API",
    description="LLM-powered intelligent query-retrieval system for insurance/legal/HR/compliance",
    version="3.1.0"
)

# ----------- API Models -----------
class AnalyzeRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class AnalyzeResponse(BaseModel):
    answers: List[str]

# ----------- Cache and config -----------
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_CACHE = {}  # doc_hash -> chain
FAISS_PATH_CACHE = {}
# concurrency limit for parallel LLM calls
DEFAULT_MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))
# embedding model names
EMBED_LARGE = "text-embedding-3-large"
EMBED_SMALL = "text-embedding-3-small"
# threshold to switch to small embeddings for large docs
LARGE_DOC_TOKEN_THRESHOLD = 500_000  # heuristic based on document text length

# ----------- Helpers -----------
def get_doc_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def normalize_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def safe_save_faiss(faiss_index: FAISS, path: str):
    faiss_index.save_local(path)

def choose_embedding_model(document_text: str) -> str:
    if len(document_text) > LARGE_DOC_TOKEN_THRESHOLD:
        return EMBED_SMALL
    return EMBED_LARGE

# ----------- Extraction (PDF/DOCX/Email) -----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text preserving simple table structure.
    Uses PyMuPDF text extraction; falls back to OCR for pages with no text.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        has_any_text = False
        for pno, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            page_lines = []
            for block in data.get("blocks", []):
                if "lines" not in block or not block["lines"]:
                    continue
                # collect x positions to detect columns
                x_positions = []
                row_cells = []
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    x_positions.append(round(spans[0]["bbox"][0], -1))
                    cell_text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                    cell_text = normalize_text(cell_text)
                    if cell_text:
                        row_cells.append(cell_text)
                unique_x = set(x_positions)
                if len(unique_x) > 1:
                    # table-like block
                    page_lines.append("[TABLE START]")
                    # emit each detected row as pipe-separated (best-effort)
                    for r in row_cells:
                        page_lines.append(" | ".join([r]))
                    page_lines.append("[TABLE END]")
                else:
                    paragraph = " ".join([r for r in row_cells if r])
                    if paragraph:
                        page_lines.append(paragraph)
            page_text = normalize_text("\n".join(page_lines))
            if page_text:
                has_any_text = True
            pages.append(f"[Page {pno}]\n{page_text}")
            # free block-level objects implicitly by overwriting variables
        doc.close()

        if not has_any_text:
            # OCR fallback: one page at a time to avoid huge memory use
            images = convert_from_bytes(file_bytes)
            ocr_pages = []
            for i, img in enumerate(images, start=1):
                ocr_text = pytesseract.image_to_string(img)
                ocr_pages.append(f"[Page {i}]\n{normalize_text(ocr_text)}")
            return "\n\n".join(ocr_pages)

        return "\n\n".join(pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        txt = docx2txt.process("temp.docx")
        return normalize_text(txt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX: {e}")

def extract_text_from_email(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        body = msg.get_body(preferencelist=('plain', 'html'))
        return normalize_text(body.get_content()) if body else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process email: {e}")

def detect_file_type_and_extract(url: str) -> str:
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        file_bytes = resp.content
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_text_from_pdf(file_bytes)
        if "wordprocessingml" in content_type or url.lower().endswith(".docx"):
            return extract_text_from_docx(file_bytes)
        if "message" in content_type or content_type == "application/octet-stream":
            return extract_text_from_email(file_bytes)
        # fallback try PDF parser
        return extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch/extract document: {e}")

# ----------- Prompt templates (strict final prompt) -----------
RETRIEVE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a policy/contract/HR/compliance assistant.
Use ONLY the following excerpts from the document:
{context}

Question: {question}

Provide a concise, direct answer based strictly on the clauses above.
Avoid assumptions. If the context does not explicitly answer the question, return "NOT FOUND".
"""
)

FINAL_STRICT_PROMPT_TEMPLATE = """
You are an expert policy/legal/HR/compliance assistant.
Use ONLY the CONTEXT excerpts below to answer the QUESTION. Do NOT use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS (must follow exactly):
1. If the context contains a sentence/clause that answers the question, QUOTE that clause verbatim (include numbers and parentheses exactly).
2. If two nearby clauses together answer, quote both clauses and join them with a single space.
3. If no clause answers the question, respond exactly with: "NOT FOUND"
4. Output valid JSON ONLY with this shape:
   {{ "answer": "<one concise sentence or NOT FOUND>", "sources": ["chunk_1","chunk_2"] }}
5. The "answer" string must be one concise sentence (no extra commentary).

Now produce the JSON only.
"""

# ----------- FAISS / embedding utilities -----------
def build_faiss_from_documents(documents: List[Document], embed_model_name: str) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=embed_model_name,
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
    )
    return FAISS.from_documents(documents, embeddings)

def save_and_release_faiss(faiss_index: FAISS, path: str):
    faiss_index.save_local(path)
    # free the object; we'll reload it for serving
    del faiss_index
    gc.collect()

def load_faiss_readonly(path: str, embed_model_name: str) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=embed_model_name,
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_retrieval_chain(faiss_index: FAISS, return_source_documents: bool = True, k: int = 12):
    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_index.as_retriever(search_kwargs={"k": k, "search_type": "mmr"}),
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": RETRIEVE_PROMPT}
    )

# ----------- Build or load chain with caching (memory mindful) -----------
def get_chain_with_cache(document_text: str):
    """
    Build FAISS on-disk index if missing, otherwise load read-only.
    Use adaptive embedding model based on document length.
    """
    doc_hash = get_doc_hash(document_text)
    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]

    cache_path = os.path.join(CACHE_DIR, doc_hash)
    embed_model = choose_embedding_model(document_text)

    if os.path.exists(cache_path):
        faiss_index = load_faiss_readonly(cache_path, embed_model)
        chain = create_retrieval_chain(faiss_index, return_source_documents=True, k=12)
        INDEX_CACHE[doc_hash] = chain
        FAISS_PATH_CACHE[doc_hash] = cache_path
        return chain

    # chunk document into clause-friendly segments
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,    # smaller chunks make clauses more focused
        chunk_overlap=400,  # larger overlap avoids split clauses
        separators=["\n\n", "\n", ". ", " "]
    )
    # create Document objects with metadata
    raw_chunks = splitter.split_text(document_text)
    documents = [Document(page_content=txt, metadata={"chunk_id": idx + 1,
                                                      "is_table": ("[TABLE START]" in txt or "|" in txt)})
                 for idx, txt in enumerate(raw_chunks)]
    # build FAISS with chosen embedding model
    faiss_index = build_faiss_from_documents(documents, embed_model)
    # save to disk and release memory
    save_and_release_faiss(faiss_index, cache_path)
    # reload read-only
    faiss_index = load_faiss_readonly(cache_path, embed_model)
    chain = create_retrieval_chain(faiss_index, return_source_documents=True, k=12)
    INDEX_CACHE[doc_hash] = chain
    FAISS_PATH_CACHE[doc_hash] = cache_path
    # free large locals
    del documents, raw_chunks
    gc.collect()
    return chain

# ----------- Two-step answer logic with fallback retrieval -----------
async def final_strict_answer_from_context(context_chunks: List[str], question: str, sources_labels: List[str]) -> dict:
    """
    Sends strict final prompt to LLM to produce JSON-only answer quoting verbatim or NOT FOUND.
    context_chunks: list of text strings (already prioritized)
    sources_labels: list of labels matching context_chunks
    """
    # Limit context length (join up to first N chunks)
    MAX_CHUNKS_FOR_CONTEXT = 8
    used_chunks = context_chunks[:MAX_CHUNKS_FOR_CONTEXT]
    used_sources = sources_labels[:MAX_CHUNKS_FOR_CONTEXT]
    context = "\n\n---\n\n".join(used_chunks)

    prompt = FINAL_STRICT_PROMPT_TEMPLATE.format(context=context, question=question)

    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    # run LLM
    raw = await asyncio.to_thread(llm.predict, prompt)
    raw = raw.strip()
    # strip common fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()

    # try parse JSON
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed.get("answer"), str):
            return {"answer": "NOT FOUND", "sources": used_sources}
        if not isinstance(parsed.get("sources"), list):
            parsed["sources"] = used_sources
        return parsed
    except Exception:
        # if not pure JSON, extract first JSON-like substring
        m = re.search(r"(\{.*\})", raw, flags=re.S)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return parsed
            except Exception:
                return {"answer": "NOT FOUND", "sources": used_sources}
        return {"answer": "NOT FOUND", "sources": used_sources}

async def ask_question_with_retrieval(chain, question: str, semaphore: asyncio.Semaphore, fallback_k: int = 20) -> dict:
    """
    1) Use chain to retrieve source_documents (k=12 MMR)
    2) Build prioritized context (tables first)
    3) Call strict final prompt LLM to quote verbatim or NOT FOUND
    4) If answer == NOT FOUND, perform fallback retrieval with higher k and retry once.
    Returns parsed JSON dict {answer, sources}
    """
    async with semaphore:
        try:
            # Step A: retrieval with chain (this returns a dict with source_documents)
            raw = await asyncio.to_thread(chain, {"query": question})
            # normalize possible return structures
            docs = []
            if isinstance(raw, dict):
                docs = raw.get("source_documents") or raw.get("source_documents", []) or []
            elif hasattr(raw, "source_documents"):
                docs = getattr(raw, "source_documents", []) or []
            else:
                # fallback: try retriever directly if chain has retriever
                retriever = getattr(chain, "retriever", None)
                if retriever:
                    docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
            if not docs:
                return {"answer": "NOT FOUND", "sources": []}

            # prioritize table chunks
            docs_sorted = sorted(docs, key=lambda d: 0 if ((getattr(d, "metadata", {}) or d.get("metadata", {})).get("is_table")) else 1)

            context_chunks = []
            sources = []
            for d in docs_sorted:
                txt = getattr(d, "page_content", "") or d.get("page_content", "")
                if not txt:
                    continue
                # small normalization: remove excessive page markers but keep content
                context_chunks.append(txt)
                md = getattr(d, "metadata", {}) or d.get("metadata", {})
                sources.append(f"chunk_{md.get('chunk_id', md.get('page', 'unknown'))}")

            # Final strict LLM call
            final_resp = await final_strict_answer_from_context(context_chunks, question, sources)
            if isinstance(final_resp, dict) and final_resp.get("answer") != "NOT FOUND":
                return final_resp

            # Fallback: perform broader retrieval with higher k (only once)
            # Build a temporary retriever from chain.retriever if accessible
            retriever = getattr(chain, "retriever", None)
            if retriever:
                # temporarily perform a get_relevant_documents with higher k
                try:
                    more_docs = await asyncio.to_thread(retriever.get_relevant_documents, question, search_kwargs={"k": fallback_k, "search_type": "mmr"})
                except TypeError:
                    # some retriever implementations ignore kwargs here; call default with more results via chain
                    more_raw = await asyncio.to_thread(chain, {"query": question})
                    more_docs = more_raw.get("source_documents") or []
                if more_docs:
                    # sort and rebuild context, then retry final strict prompt
                    more_sorted = sorted(more_docs, key=lambda d: 0 if ((getattr(d, "metadata", {}) or d.get("metadata", {})).get("is_table")) else 1)
                    more_chunks = []
                    more_sources = []
                    for d in more_sorted:
                        txt = getattr(d, "page_content", "") or d.get("page_content", "")
                        if not txt:
                            continue
                        more_chunks.append(txt)
                        md = getattr(d, "metadata", {}) or d.get("metadata", {})
                        more_sources.append(f"chunk_{md.get('chunk_id', md.get('page', 'unknown'))}")
                    final_retry = await final_strict_answer_from_context(more_chunks, question, more_sources)
                    return final_retry
            # nothing else - return NOT FOUND
            return {"answer": "NOT FOUND", "sources": sources}
        except Exception as e:
            return {"answer": f"ERROR: {str(e)}", "sources": []}

# ----------- Parallel batch processing -----------
async def process_questions_parallel(chain, questions: List[str], max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [ask_question_with_retrieval(chain, q, semaphore) for q in questions]
    results = await asyncio.gather(*tasks)
    return results

# ----------- API endpoints (same signatures) -----------
@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest):
    # Step 1: extract document text
    document_text = detect_file_type_and_extract(req.documents)
    if not document_text:
        raise HTTPException(status_code=400, detail="Empty document")
    # Step 2: get or build chain with cached FAISS
    chain = get_chain_with_cache(document_text)

    # Step 3: process questions in parallel (two-step retrieval+strict LLM)
    max_conc = DEFAULT_MAX_CONCURRENCY
    try:
        max_conc = int(os.getenv("MAX_CONCURRENCY", str(DEFAULT_MAX_CONCURRENCY)))
    except Exception:
        max_conc = DEFAULT_MAX_CONCURRENCY

    parsed_results = await process_questions_parallel(chain, req.questions, max_concurrency=max_conc)

    # convert parsed JSON dicts to answer strings (the original API expected list[str])
    answers = []
    for pr in parsed_results:
        if not isinstance(pr, dict):
            answers.append("NOT FOUND")
            continue
        ans = pr.get("answer", "NOT FOUND")
        # ensure single-line answer
        if isinstance(ans, str):
            answers.append(ans.strip().replace("\n", " "))
        else:
            answers.append("NOT FOUND")
    return AnalyzeResponse(answers=answers)

@app.get("/")
def root():
    return {"message": "AI Insurance Document Analyzer is running"}

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)

# End of file
