import csv
import io
from fastapi.responses import StreamingResponse
import json
from pathlib import Path
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import Optional
from tasks import process_document
from models.schemas import UploadResponse, ExtractionResult, DocumentStatus
from services.ocr import extract_text_from_file
from services.extractor import extract_structured_data
from services.storage import init_storage, upload_file, get_file_url
from database.connection import init_db, get_db
from database import crud

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── After all imports ─────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add QARequest here — before app creation
class QARequest(BaseModel):
    question: str
    doc_id: Optional[str] = None


# ── Startup / Shutdown ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Startup  → create DB tables + MinIO bucket
    Shutdown → nothing needed
    """
    logger.info("Starting up DocParse API...")
    await init_db()        # create tables if not exist
    init_storage()         # create MinIO bucket if not exist
    logger.info("Database and storage ready.")
    yield
    logger.info("Shutting down DocParse API...")


# ── FastAPI app ───────────────────────────────────
app = FastAPI(
    title="DocParse API",
    description="Intelligent Document Processing Pipeline",
    version="2.0.0",
    lifespan=lifespan
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ── Health check ──────────────────────────────────
@app.get("/")
def root():
    return {"message": "DocParse API is running", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Upload endpoint ───────────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF, PNG, JPG or TIFF."
        )

    # Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    # Create DB record immediately — status = processing
    doc = await crud.create_document(
        db=db,
        filename=file.filename,
        content_type=file.content_type,
        s3_key=""
    )
    doc_id = str(doc.id)
    logger.info(f"Created document: {doc_id}")

    # Send to Celery — returns INSTANTLY
    
    process_document.delay(
        file_bytes.hex(),
        file.filename,
        file.content_type,
        doc_id
    )

    # Return immediately without waiting
    return UploadResponse(
        doc_id=doc_id,
        status=DocumentStatus.processing,
        message="Document queued for processing"
    )



# ── Get results endpoint ──────────────────────────
@app.get("/results/{doc_id}", response_model=ExtractionResult)
async def get_results(
    doc_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get extraction results for a document by its ID.
    Reads from PostgreSQL — persists across restarts.
    """
    try:
        uid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await crud.get_document(db, uid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    result = doc.result
    if not result:
        raise HTTPException(status_code=404, detail="Result not ready yet")

    return ExtractionResult(
        doc_id=str(doc.id),
        status=DocumentStatus(doc.status),
        document_type=result.document_type,
        extracted_data=result.extracted_data,
        confidence=result.confidence,
        raw_text=result.raw_text,
        error=result.error
    )


# ── List all documents endpoint ───────────────────
@app.get("/documents")
async def list_documents(
    db: AsyncSession = Depends(get_db)
):
    """
    List all uploaded documents with their status.
    Reads from PostgreSQL.
    """
    docs = await crud.get_all_documents(db)
    return [
        {
            "doc_id":        str(d.id),
            "filename":      d.filename,
            "status":        d.status,
            "created_at":    d.created_at.isoformat(),
            "download_url":  get_file_url(d.s3_key) if d.s3_key else None
        }
        for d in docs
    ]


# ── Get single document with download URL ─────────
@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        uid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    doc = await crud.get_document(db, uid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id":       str(doc.id),
        "filename":     doc.filename,
        "status":       doc.status,
        "created_at":   doc.created_at.isoformat(),
        "download_url": get_file_url(doc.s3_key) if doc.s3_key else None
    }


@app.get("/benchmark")
def get_benchmark():
    path = Path(__file__).parent / "benchmarks.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No benchmark data yet.")
    with open(path) as f:
        return json.load(f)


@app.post("/batch-upload")
async def batch_upload(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    results = []
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            results.append({"filename": file.filename, "error": "Unsupported type"})
            continue
        file_bytes = await file.read()
        doc = await crud.create_document(db, file.filename, file.content_type, "")
        try:
            s3_key = upload_file(file_bytes, file.filename, file.content_type, str(doc.id))
            doc.s3_key = s3_key
            raw_text = extract_text_from_file(file_bytes, file.filename)
            extraction = extract_structured_data(raw_text)
            await crud.create_extraction_result(
                db, doc.id,
                extraction.get("document_type", "unknown"),
                extraction.get("extracted_data", {}),
                extraction.get("confidence", {}),
                raw_text
            )
            await crud.update_document_status(db, doc.id, "done")
            results.append({
                "filename": file.filename,
                "doc_id": str(doc.id),
                "status": "done",
                "document_type": extraction.get("document_type")
            })
        except Exception as e:
            await crud.update_document_status(db, doc.id, "error")
            results.append({"filename": file.filename, "error": str(e)})
    return {"total": len(files), "results": results}


@app.get("/export/csv")
async def export_csv(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    from database.models import Document
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Document).options(selectinload(Document.result))
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "doc_id", "filename", "status", "document_type",
        "invoice_number", "vendor_name", "total_amount",
        "currency", "issue_date", "due_date", "created_at"
    ])
    for doc in docs:
        r = doc.result
        data = r.extracted_data if r else {}
        writer.writerow([
            str(doc.id), doc.filename, doc.status,
            r.document_type if r else "",
            data.get("invoice_number", ""),
            data.get("vendor_name", ""),
            data.get("total_amount", ""),
            data.get("currency", ""),
            data.get("issue_date", ""),
            data.get("due_date", ""),
            doc.created_at.strftime("%Y-%m-%d %H:%M")
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=docparse_export.csv"}
    )

# ── Q&A endpoints ─────────────────────────────────

class QARequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

@app.post("/qa", include_in_schema=True)
async def question_answer(request: QARequest):
    """
    Answer a question.
    If doc_id provided → RAG on that document.
    If no doc_id → general knowledge answer.
    """
    from services.rag import answer_with_document, answer_general

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        if request.doc_id:
            result = answer_with_document(request.doc_id, request.question)
        else:
            result = answer_general(request.question)
        return result

    except Exception as e:
        logger.error(f"Q&A failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qa/index/{doc_id}")
async def index_document_endpoint(
    doc_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger indexing for an existing document."""
    from services.rag import index_document
    try:
        uid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc = await crud.get_document(db, uid)
    if not doc or not doc.result:
        raise HTTPException(status_code=404, detail="Document or result not found")

    index_document(doc_id, doc.result.raw_text)
    return {"message": f"Document {doc_id} indexed successfully"}