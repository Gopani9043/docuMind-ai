from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum

class DocumentType(str, Enum):
    invoice  = "invoice"
    contract = "contract"
    receipt  = "receipt"
    report   = "report"
    unknown  = "unknown"

class DocumentStatus(str, Enum):
    processing = "processing"
    done       = "done"
    error      = "error"

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    issue_date:     Optional[str] = None
    due_date:       Optional[str] = None
    vendor_name:    Optional[str] = None
    total_amount:   Optional[float] = None
    currency:       Optional[str] = None
    vat_amount:     Optional[float] = None
    line_items:     Optional[list] = []

class ContractData(BaseModel):
    parties:        Optional[list] = []
    start_date:     Optional[str] = None
    end_date:       Optional[str] = None
    value:          Optional[float] = None
    currency:       Optional[str] = None
    key_clauses:    Optional[list] = []

class ReceiptData(BaseModel):
    vendor_name:    Optional[str] = None
    date:           Optional[str] = None
    total_amount:   Optional[float] = None
    currency:       Optional[str] = None
    items:          Optional[list] = []

class ExtractionResult(BaseModel):
    doc_id:          str
    status:          DocumentStatus
    document_type:   DocumentType
    extracted_data:  Optional[Dict[str, Any]] = None
    confidence:      Optional[Dict[str, float]] = {}
    raw_text:        Optional[str] = None
    error:           Optional[str] = None

class UploadResponse(BaseModel):
    doc_id:  str
    status:  DocumentStatus
    message: str