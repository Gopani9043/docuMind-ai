import sys
import json
from pathlib import Path
from unittest.mock import patch

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

MOCK_LLM_RESPONSE = json.dumps({
    "document_type": "invoice",
    "extracted_data": {
        "invoice_number": "INV-2024-042",
        "issue_date": "15 October 2024",
        "due_date": "14 November 2024",
        "vendor_name": "Mueller GmbH",
        "customer_name": None,
        "total_amount": 5735.80,
        "currency": "EUR",
        "vat_amount": 915.80,
        "line_items": []
    },
    "confidence": {}
})


@patch("services.extractor.invoke_with_fallback")
def test_extractor_returns_dict(mock_invoke):
    """Extractor should always return a dict."""
    mock_invoke.return_value = MOCK_LLM_RESPONSE
    result = extract_structured_data(SAMPLE_TEXT)
    assert isinstance(result, dict)


@patch("services.extractor.invoke_with_fallback")
def test_extractor_identifies_invoice(mock_invoke):
    """Should correctly identify invoice document type."""
    mock_invoke.return_value = MOCK_LLM_RESPONSE
    result = extract_structured_data(SAMPLE_TEXT)
    assert result.get("document_type") == "invoice"


@patch("services.extractor.invoke_with_fallback")
def test_extractor_has_extracted_data(mock_invoke):
    """Should return extracted_data dict with fields."""
    mock_invoke.return_value = MOCK_LLM_RESPONSE
    result = extract_structured_data(SAMPLE_TEXT)
    assert "extracted_data" in result
    assert isinstance(result["extracted_data"], dict)


@patch("services.extractor.invoke_with_fallback")
def test_extractor_finds_vendor(mock_invoke):
    """Should extract vendor name from invoice text."""
    mock_invoke.return_value = MOCK_LLM_RESPONSE
    result = extract_structured_data(SAMPLE_TEXT)
    data = result.get("extracted_data", {})
    vendor = data.get("vendor_name", "")
    assert "Mueller" in str(vendor)


@patch("services.extractor.invoke_with_fallback")
def test_extractor_finds_total(mock_invoke):
    """Should extract total amount."""
    mock_invoke.return_value = MOCK_LLM_RESPONSE
    result = extract_structured_data(SAMPLE_TEXT)
    data = result.get("extracted_data", {})
    assert data.get("total_amount") is not None