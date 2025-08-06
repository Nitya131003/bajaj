import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import fitz  # PyMuPDF
import docx2txt
from email import policy
from email.parser import BytesParser
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Response model
class AnalysisResponse(BaseModel):
    decision: str
    sum_assured: Optional[float]
    justification: str
    retrieved_clauses: List[str]

# Initialize app
app = FastAPI(
    title="AI Insurance Document Analyzer API",
    description="Upload a policy (PDF/DOCX/Email) and ask if a medical procedure is covered.",
    version="2.0.0"
)

# ---------- File Extraction ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        return docx2txt.process("temp.docx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX: {e}")

def extract_text_from_email(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        return msg.get_body(preferencelist=('plain', 'html')).get_content()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Email: {e}")

def detect_file_type_and_extract(file: UploadFile, file_bytes: bytes) -> str:
    ctype = file.content_type or ""
    if "pdf" in ctype:
        return extract_text_from_pdf(file_bytes)
    elif "wordprocessingml" in ctype:
        return extract_text_from_docx(file_bytes)
    elif "message" in ctype or ctype == "application/octet-stream":
        return extract_text_from_email(file_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ctype}")

# ---------- LangChain Setup ----------
def create_langchain_qa_chain(chunks: List[str]):
    documents = [Document(page_content=chunk) for chunk in chunks]

    # ✅ Use Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=os.getenv("model"),  # default model
        openai_api_base=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        openai_api_type="azure"
    )

    db = FAISS.from_documents(documents, embeddings)
    
    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        openai_api_base=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        openai_api_type="azure"
    )
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)

# ---------- Query ----------
def query_with_langchain(query: str, chain) -> dict:
    try:
        result = chain(query)
        source_clauses = [doc.page_content for doc in result['source_documents']]
        combined_context = "\n\n".join(source_clauses[:3])

        llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=os.getenv("model"),
            openai_api_base=os.getenv("AZURE_API_BASE"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            openai_api_key=os.getenv("AZURE_API_KEY"),
            openai_api_type="azure"
        )

        # Decision Prompt
        decision_prompt = f"""
        You are an insurance policy expert.
        Policy excerpts:
        {combined_context}

        Query: "{query}"

        Decide if the procedure is covered based ONLY on the policy.
        Consider age, gender, waiting period, and exclusions.
        Respond in ONE word: Yes or No.
        """
        decision_raw = llm.predict(decision_prompt).strip().lower()

        # Justification Prompt
        justification_prompt = f"""
        You are an insurance policy analyst.
        Policy excerpts:
        {combined_context}

        Query: "{query}"

        Explain your decision in detail.
        Quote relevant clauses. Do NOT assume anything outside the policy.
        """
        justification_raw = llm.predict(justification_prompt).strip()

        if "no" in decision_raw:
            decision = "No, the claim is not covered under the policy"
        elif "yes" in decision_raw:
            decision = "Yes, the claim is covered under the policy"
        else:
            decision = "Unclear from the policy document. See justification."

        return {
            "decision": decision,
            "sum_assured": None,
            "justification": justification_raw,
            "retrieved_clauses": source_clauses
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain query failed: {e}")

# ---------- API Endpoint ----------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(query: str = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_text = detect_file_type_and_extract(file, file_bytes)
    chunks = [document_text[i:i + 1000] for i in range(0, len(document_text), 850)]
    chain = create_langchain_qa_chain(chunks)
    response = query_with_langchain(query, chain)
    return AnalysisResponse(**response)

@app.get("/")
def health():
    return {"status": "Running", "message": "AI Insurance Analyzer is live."}

