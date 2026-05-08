import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from services.ocr import extract_text_from_file

SAMPLES = Path(__file__).parent.parent / "sample_documents"

def test_ocr_pdf_returns_text():
    """OCR should return a non-empty string from a PDF."""
    pdf_path = SAMPLES / "invoice_test.pdf"
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    result = extract_text_from_file(file_bytes, "invoice_test.pdf")
    assert isinstance(result, str)
    assert len(result) > 50

def test_ocr_contains_expected_text():
    """OCR should find key words from our sample invoice."""
    pdf_path = SAMPLES / "invoice_test.pdf"
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    result = extract_text_from_file(file_bytes, "invoice_test.pdf")
    assert "INVOICE" in result.upper()

def test_ocr_invalid_extension_raises():
    """Unsupported file type should raise ValueError."""
    try:
        extract_text_from_file(b"fake", "file.docx")
        assert False, "Should have raised"
    except ValueError:
        pass