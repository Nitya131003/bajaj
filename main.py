import os
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
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

# ----------- Input/Output Models -----------
class AnalyzeRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class AnalyzeResponse(BaseModel):
    answers: List[str]

# ----------- App Init -----------
app = FastAPI(
    title="AI Insurance Analyzer API",
    description="Batch Q&A over insurance documents using LLMs",
    version="2.0.0"
)

# ----------- File Extraction Helpers -----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        return docx2txt.process("temp.docx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {str(e)}")

def extract_text_from_email(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        body = msg.get_body(preferencelist=('plain', 'html'))
        if body:
            return body.get_content()
        else:
            raise ValueError("Email body could not be extracted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email extraction failed: {str(e)}")

def detect_file_type_and_extract(url: str) -> str:
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch file from URL, status code {response.status_code}")
        file_bytes = response.content
        content_type = response.headers.get("Content-Type", "")

        if "pdf" in content_type:
            return extract_text_from_pdf(file_bytes)
        elif "wordprocessingml" in content_type or "docx" in content_type:
            return extract_text_from_docx(file_bytes)
        elif "message" in content_type or content_type == "application/octet-stream":
            return extract_text_from_email(file_bytes)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or parsing document: {str(e)}")

# ----------- LangChain Setup -----------
def create_langchain_chain(chunks: List[str]):
    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )

    db = FAISS.from_documents(documents, embeddings)

    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)

# ----------- Query Runner -----------
def run_batch_questions(questions: List[str], chain) -> List[str]:
    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )

    answers = []
    for query in questions:
        try:
            result = chain(query)
            context = "\n\n".join(doc.page_content for doc in result['source_documents'][:3])

            prompt = f"""
            You are a smart insurance policy assistant.
            Based ONLY on the following policy excerpts:
            {context}

            Question: "{query}"

            Answer concisely, citing clauses if needed, with no outside assumptions.
            """
            response = llm.predict(prompt).strip()
            answers.append(response)
        except Exception as e:
            answers.append(f"Error answering question: {str(e)}")

    return answers

# ----------- API Endpoint -----------
@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(
    req: AnalyzeRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization token.")

    document_text = detect_file_type_and_extract(req.documents)
    chunks = [document_text[i:i + 1000] for i in range(0, len(document_text), 850)]
    chain = create_langchain_chain(chunks)
    answers = run_batch_questions(req.questions, chain)
    return AnalyzeResponse(answers=answers)

@app.get("/")
def health():
    return {"status": "ok", "message": "AI Insurance Analyzer is running."}
