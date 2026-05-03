import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.schemas import UploadResponse, ExtractionResult, DocumentStatus
from services.ocr import extract_text_from_file
from services.extractor import extract_structured_data

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocParse API",
    description="Intelligent Document Processing Pipeline",
    version="1.0.0"
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary in-memory store (we replace with PostgreSQL in Phase 2)
results_store: dict = {}

ALLOWED_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

@app.get("/")
def root():
    return {"message": "DocParse API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF/image) → OCR → LLM extraction → return doc_id
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF, PNG, JPG or TIFF."
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    logger.info(f"Processing document: {file.filename} [{doc_id}]")

    try:
        # Step 1: OCR — extract raw text
        logger.info("Step 1/2: Running OCR...")
        raw_text = extract_text_from_file(file_bytes, file.filename)

        # Step 2: LLM — extract structured data
        logger.info("Step 2/2: Running LLM extraction...")
        extraction = extract_structured_data(raw_text)

        # Store result
        results_store[doc_id] = {
            "doc_id": doc_id,
            "status": DocumentStatus.done,
            "document_type": extraction.get("document_type", "unknown"),
            "extracted_data": extraction.get("extracted_data", {}),
            "confidence": extraction.get("confidence", {}),
            "raw_text": raw_text,
            "filename": file.filename,
        }

        return UploadResponse(
            doc_id=doc_id,
            status=DocumentStatus.done,
            message=f"Document processed successfully as {extraction.get('document_type')}"
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        results_store[doc_id] = {
            "doc_id": doc_id,
            "status": DocumentStatus.error,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{doc_id}", response_model=ExtractionResult)
def get_results(doc_id: str):
    """Get extraction results for a document by its ID."""
    if doc_id not in results_store:
        raise HTTPException(status_code=404, detail="Document not found")
    return results_store[doc_id]

@app.get("/documents")
def list_documents():
    """List all processed documents."""
    return [
        {
            "doc_id": v["doc_id"],
            "filename": v.get("filename", "unknown"),
            "status": v["status"],
            "document_type": v.get("document_type", "unknown"),
        }
        for v in results_store.values()
    ]