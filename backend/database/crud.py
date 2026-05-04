import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from database.models import Document, ExtractionResult


# ─────────────────────────────────────────────────
# DOCUMENT FUNCTIONS
# ─────────────────────────────────────────────────

async def create_document(
    db: AsyncSession,
    filename: str,
    content_type: str,
    s3_key: str
) -> Document:
    """
    Insert a new document row when a file is uploaded.
    Status starts as 'processing'.
    """
    doc = Document(
        filename=filename,
        content_type=content_type,
        status="processing",
        s3_key=s3_key
    )
    db.add(doc)
    await db.flush()   # gets the generated UUID without committing yet
    return doc


async def update_document_status(
    db: AsyncSession,
    doc_id: uuid.UUID,
    status: str
) -> None:
    """
    Update status to 'done' or 'error' after processing.
    """
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    doc = result.scalar_one_or_none()
    if doc:
        doc.status = status
        await db.flush()


async def get_document(
    db: AsyncSession,
    doc_id: uuid.UUID
) -> Document | None:
    """
    Fetch a single document by its ID.
    Also loads the related extraction result in one query.
    """
    result = await db.execute(
        select(Document)
        .where(Document.id == doc_id)
        .options(selectinload(Document.result))
    )
    return result.scalar_one_or_none()


async def get_all_documents(
    db: AsyncSession
) -> list[Document]:
    """
    Fetch all documents ordered by newest first.
    Used by GET /documents endpoint.
    """
    result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────
# EXTRACTION RESULT FUNCTIONS
# ─────────────────────────────────────────────────

async def create_extraction_result(
    db: AsyncSession,
    doc_id: uuid.UUID,
    document_type: str,
    extracted_data: dict,
    confidence: dict,
    raw_text: str
) -> ExtractionResult:
    """
    Save the LLM extraction result linked to a document.
    """
    extraction = ExtractionResult(
        doc_id=doc_id,
        document_type=document_type,
        extracted_data=extracted_data,
        confidence=confidence,
        raw_text=raw_text
    )
    db.add(extraction)
    await db.flush()
    return extraction


async def create_error_result(
    db: AsyncSession,
    doc_id: uuid.UUID,
    error_message: str
) -> ExtractionResult:
    """
    Save an error result when processing fails.
    """
    extraction = ExtractionResult(
        doc_id=doc_id,
        document_type="unknown",
        extracted_data={},
        confidence={},
        raw_text=None,
        error=error_message
    )
    db.add(extraction)
    await db.flush()
    return extraction


async def get_extraction_result(
    db: AsyncSession,
    doc_id: uuid.UUID
) -> ExtractionResult | None:
    """
    Fetch extraction result by document ID.
    """
    result = await db.execute(
        select(ExtractionResult)
        .where(ExtractionResult.doc_id == doc_id)
    )
    return result.scalar_one_or_none()