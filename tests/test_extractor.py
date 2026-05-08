import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from services.extractor import extract_structured_data

SAMPLE_TEXT = """
INVOICE
Invoice Number : INV-2024-042
Issue Date     : 15 October 2024
Due Date       : 14 November 2024
Vendor: Mueller GmbH
Total Due: EUR 5735.80
VAT 19%: EUR 915.80
"""

def test_extractor_returns_dict():
    """Extractor should always return a dict."""
    result = extract_structured_data(SAMPLE_TEXT)
    assert isinstance(result, dict)

def test_extractor_identifies_invoice():
    """Should correctly identify invoice document type."""
    result = extract_structured_data(SAMPLE_TEXT)
    assert result.get("document_type") == "invoice"

def test_extractor_has_extracted_data():
    """Should return extracted_data dict with fields."""
    result = extract_structured_data(SAMPLE_TEXT)
    assert "extracted_data" in result
    assert isinstance(result["extracted_data"], dict)

def test_extractor_finds_vendor():
    """Should extract vendor name from invoice text."""
    result = extract_structured_data(SAMPLE_TEXT)
    data = result.get("extracted_data", {})
    vendor = data.get("vendor_name", "")
    assert "Mueller" in str(vendor)

def test_extractor_finds_total():
    """Should extract total amount."""
    result = extract_structured_data(SAMPLE_TEXT)
    data = result.get("extracted_data", {})
    assert data.get("total_amount") is not None