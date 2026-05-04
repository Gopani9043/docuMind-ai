import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database.connection import Base


class Document(Base):
    """
    One row per uploaded document.
    Stores filename, type, status and when it was created.
    """
    __tablename__ = "documents"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename     = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    status       = Column(String(50), default="processing")
    s3_key       = Column(String(500), nullable=True)   # path inside MinIO bucket
    created_at   = Column(DateTime, default=datetime.utcnow)

    # One document → one result (one-to-one relationship)
    result = relationship("ExtractionResult", back_populates="document", uselist=False)


class ExtractionResult(Base):
    """
    One row per extraction result.
    Linked to a Document by doc_id foreign key.
    """
    __tablename__ = "extraction_results"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id           = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    document_type    = Column(String(50), default="unknown")
    extracted_data   = Column(JSON, nullable=True)       # stores the full extracted JSON
    confidence       = Column(JSON, nullable=True)       # stores confidence scores
    raw_text         = Column(Text, nullable=True)       # full OCR text
    error            = Column(Text, nullable=True)       # error message if failed
    created_at       = Column(DateTime, default=datetime.utcnow)

    # Link back to Document
    document = relationship("Document", back_populates="result")