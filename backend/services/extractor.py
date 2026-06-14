import os
import re
import json
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langdetect import detect

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
)

ENGLISH_PROMPT = ChatPromptTemplate.from_template("""
You are an expert document extraction AI.
Identify document type: invoice, contract, receipt, report, unknown.
Extract ALL relevant fields based on type.

For INVOICE: invoice_number, issue_date, due_date, vendor_name, total_amount, currency, vat_amount, line_items
For CONTRACT: parties, start_date, end_date, value, currency, key_clauses
For RECEIPT: vendor_name, date, total_amount, currency, items
For REPORT: title, date, author, key_findings, summary

IMPORTANT EXTRACTION RULES:
- total_amount: extract as pure number only — no currency symbols, no commas
  CORRECT: 5735.80
  WRONG: €5,735.80 or 5.735,80 or $1,234.56
- currency: extract as 3-letter code only — EUR, USD, GBP, CHF, JPY, INR
- If amount uses European format (1.234,56) convert to standard (1234.56)

Return ONLY valid JSON:
{{
  "document_type": "invoice",
  "extracted_data": {{}},
  "confidence": {{}}
}}

DOCUMENT:
{text}
""")

GERMAN_PROMPT = ChatPromptTemplate.from_template("""
Du bist ein KI-Experte für Dokumentenextraktion.
Identifiziere den Dokumenttyp: invoice, contract, receipt, report, unknown.
Extrahiere ALLE relevanten Felder.

Für RECHNUNG (invoice): invoice_number, issue_date, due_date, vendor_name, total_amount, currency, vat_amount, line_items
Für VERTRAG (contract): parties, start_date, end_date, value, currency, key_clauses
Für KASSENBON (receipt): vendor_name, date, total_amount, currency, items
Für BERICHT (report): title, date, author, key_findings, summary

WICHTIG:
- total_amount: NUR als reine Zahl — keine Währungssymbole, keine Punkte als Tausendertrennzeichen
  RICHTIG: 5735.80
  FALSCH: €5.735,80 oder 5.735,80
- currency: NUR als 3-Buchstaben-Code — EUR, USD, GBP usw.
- Antworte NUR mit gültigem JSON. Kein Text davor oder danach.
- Feldnamen IMMER auf Englisch.

Antworte in diesem Format:
{{"document_type": "invoice", "extracted_data": {{}}, "confidence": {{}}}}

DOKUMENT:
{text}
""")


# ── Amount cleaning ───────────────────────────────────────────────────────────

def clean_amount(value) -> str:
    """
    Clean extracted amount values before saving.
    Removes currency symbols and thousands separators.
    Handles both US format (1,234.56) and European format (1.234,56).

    Examples:
        €5,735.80  → 5735.80
        $1,234.56  → 1234.56
        5.735,80   → 5735.80  (European)
        ¥130,052   → 130052
        1 234.56   → 1234.56  (space separator)
    """
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""

    # Remove currency symbols
    value = re.sub(r'[€$£¥₹฿₩₪₫₭₮]', '', value)
    value = value.strip()

    # Detect European format: 1.234,56 (dot as thousands, comma as decimal)
    if re.search(r'^\d{1,3}(\.\d{3})+,\d{1,2}$', value):
        value = value.replace('.', '').replace(',', '.')
    else:
        # US/standard format: remove commas and spaces used as thousands separators
        value = value.replace(',', '').replace(' ', '')

    # Remove any remaining non-numeric chars except dot and minus
    value = re.sub(r'[^\d.\-]', '', value)

    return value.strip()


def clean_extracted_data(data: dict, document_type: str) -> dict:
    """
    Clean all numeric fields in extracted data.
    Called after LLM extraction — universal for any document type.
    """
    if not data:
        return data

    # Fields that should be clean numeric values
    amount_fields = [
        "total_amount", "vat_amount", "value",
        "subtotal", "tax_amount", "discount"
    ]

    for field in amount_fields:
        if field in data and data[field] is not None:
            original = data[field]
            cleaned = clean_amount(data[field])
            if cleaned != str(original):
                logger.info(f"Cleaned {field}: '{original}' → '{cleaned}'")
            data[field] = cleaned

    # Clean line items if present
    if "line_items" in data and isinstance(data["line_items"], list):
        for item in data["line_items"]:
            if isinstance(item, dict):
                for field in ["amount", "price", "total", "unit_price", "quantity"]:
                    if field in item and item[field] is not None:
                        item[field] = clean_amount(item[field])

    return data


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language of text. Returns 'de' for German, 'en' for others."""
    try:
        lang = detect(text[:1000])
        logger.info(f"Detected language: {lang}")
        return lang
    except Exception:
        return "en"


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_structured_data(raw_text: str) -> dict:
    try:
        lang = detect_language(raw_text)
        prompt = GERMAN_PROMPT if lang == "de" else ENGLISH_PROMPT

        chain = prompt | llm
        response = chain.invoke({"text": raw_text[:8000]})
        content = response.content.strip()

        # Strip markdown fences
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]

        content = content.strip()

        if not content:
            logger.warning("LLM returned empty content")
            return {
                "document_type": "unknown",
                "extracted_data": {},
                "confidence": {},
                "language": lang
            }

        result = json.loads(content)
        result["language"] = lang

        # ── Clean numeric fields after extraction ──
        if "extracted_data" in result and result["extracted_data"]:
            result["extracted_data"] = clean_extracted_data(
                result["extracted_data"],
                result.get("document_type", "unknown")
            )

        logger.info(f"Extraction successful: {result.get('document_type')} [{lang}]")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {
            "document_type": "unknown",
            "extracted_data": {},
            "confidence": {},
            "language": "unknown"
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise