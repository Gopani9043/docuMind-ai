import os
import re
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
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

For INVOICE: invoice_number, issue_date, due_date, vendor_name, customer_name, total_amount, currency, vat_amount, line_items
For CONTRACT: parties, start_date, end_date, value, currency, key_clauses
For RECEIPT: vendor_name, date, total_amount, currency, items
For REPORT: title, date, author, key_findings, summary

VENDOR VS CUSTOMER — FOLLOW THIS DECISION TREE EXACTLY:

STEP 1 — Find the BILL TO / SHIP TO section:
  → Whatever name appears after "BILL TO:", "SHIP TO:", "INVOICE TO:",
    "CLIENT:", "CUSTOMER:", "ATTN:", "RECIPIENT:" → that is customer_name
  → NEVER put this name as vendor_name

STEP 2 — Find the sender/issuer:
  → The company name at the TOP of the document (letterhead, header, logo area)
  → OR after "FROM:", "ISSUED BY:", "SELLER:", "SUPPLIER:"
  → OR the company whose bank account / payment details appear at the bottom
  → That is vendor_name

STEP 3 — Verify your answer:
  → vendor_name = who RECEIVES the money (has bank account on invoice)
  → customer_name = who PAYS the money (has "Bill To" label)
  → If same name appears in both positions → set vendor_name = null

STEP 4 — If still unsure:
  → vendor_name = null (never guess)
  → customer_name = null (never guess)

REAL EXAMPLES:
  "ACME Corp\n\nBILL TO: TechStart GmbH"
  → vendor_name = "ACME Corp", customer_name = "TechStart GmbH" ✓

  "BILL TO: John Smith\nFrom: Global Supplies Ltd"
  → vendor_name = "Global Supplies Ltd", customer_name = "John Smith" ✓

  "Invoice #001\nBILL TO: Acme Corp\nBank: IBAN DE89..."
  → vendor_name = null (issuer unclear), customer_name = "Acme Corp" ✓

AMOUNT RULES:
- total_amount: pure number only — no symbols, no commas
  CORRECT: 5735.80
  WRONG: €5,735.80 or 5.735,80
- currency: 3-letter code only — EUR, USD, GBP, CHF, JPY, INR
- European format (1.234,56) → convert to (1234.56)

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
Du bist eine KI, die auf die Extraktion von Daten aus Dokumenten spezialisiert ist.
Identifiziere den Dokumententyp: Rechnung, Vertrag, Quittung, Bericht, unbekannt.
Extrahiere alle relevanten Felder basierend auf dem Typ.

Für RECHNUNG: invoice_number, issue_date, due_date, vendor_name, customer_name, total_amount, currency, vat_amount, line_items
Für VERTRAG: parties, start_date, end_date, value, currency, key_clauses
Für QUITTUNG: vendor_name, date, total_amount, currency, items
Für BERICHT: title, date, author, key_findings, summary

LIEFERANT (VENDOR) VS. KUNDE (CUSTOMER) — BEFOLGE DIESEN ENTSCHEIDUNGSBAUM GENAU:

SCHRITT 1 — Suche den Bereich „BILL TO“ (Rechnungsempfänger) / „SHIP TO“ (Lieferadresse):
→ Welcher Name auch immer nach „BILL TO:“, „SHIP TO:“, „INVOICE TO:“,
„CLIENT:“, „CUSTOMER:“, „ATTN:“ oder „RECIPIENT:“ steht → das ist customer_name
→ Setze diesen Namen NIEMALS als vendor_name ein

SCHRITT 2 — Suche den Absender/Aussteller:
→ Der Firmenname ganz OBEN im Dokument (Briefkopf, Kopfzeile, Logo-Bereich)
→ ODER nach „FROM:“, „ISSUED BY:“, „SELLER:“ oder „SUPPLIER:“
→ ODER das Unternehmen, dessen Bankverbindung / Zahlungsdaten unten aufgeführt sind
→ Das ist vendor_name

SCHRITT 3 — Überprüfe deine Antwort:
→ vendor_name = wer das Geld ERHÄLT (dessen Bankkonto auf der Rechnung steht)
→ customer_name = wer das Geld BEZAHLT (bei dem die Bezeichnung „Bill To“ steht)
→ Wenn derselbe Name an beiden Positionen steht → setze vendor_name = null

SCHRITT 4 — Falls weiterhin unsicher:
→ vendor_name = null (niemals raten)
→ customer_name = null (niemals raten)

ECHTE BEISPIELE:
"ACME Corp\n\nBILL TO: TechStart GmbH"
→ vendor_name = "ACME Corp", customer_name = "TechStart GmbH" ✓

"BILL TO: John Smith\nFrom: Global Supplies Ltd"
→ vendor_name = "Global Supplies Ltd", customer_name = "John Smith" ✓

"Rechnung #001\nRECHNUNGSEMPFÄNGER: Acme Corp\nBank: IBAN DE89..."
→ vendor_name = null (Aussteller unklar), customer_name = "Acme Corp" ✓

REGELN FÜR BETRÄGE:
- total_amount: nur reine Zahl — keine Symbole, keine Kommas
KORREKT: 5735.80
FALSCH: €5,735.80 oder 5.735,80
- Währung: nur 3-Buchstaben-Code — EUR, USD, GBP, CHF, JPY, INR
- Europäisches Format (1.234,56) → in (1234.56) umwandeln

Geben Sie NUR gültiges JSON zurück:
{{
"document_type": "invoice",
"extracted_data": {{}},
"confidence": {{}}
}}

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

def fix_vendor_customer_swap(extracted_data: dict, raw_text: str) -> dict:
    """
    Detect and fix vendor/customer swap.
    If vendor_name matches a BILL TO name in the raw text → move to customer_name.
    """
    if not extracted_data or not raw_text:
        return extracted_data

    vendor_name = extracted_data.get("vendor_name") or ""
    if not vendor_name:
        return extracted_data

    bill_to_patterns = [
        r'BILL\s+TO\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'INVOICE\s+TO\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'SHIP\s+TO\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'CLIENT\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'CUSTOMER\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'RECIPIENT\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'ATTN\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'BILL\s+TO\s*:\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'INVOICE\s+TO\s*:\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'RECHNUNGSEMPFÄNGER\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'AN\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
        r'KUNDE\s*:?\s*\n+\s*([A-Za-z][A-Za-z0-9\s&.,\-]+?)(?:\n|$)',
    ]

    bill_to_names = []
    for pattern in bill_to_patterns:
        matches = re.findall(pattern, raw_text, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            if name and len(name) > 2:
                bill_to_names.append(name.lower())

    if not bill_to_names:
        return extracted_data

    vendor_lower = vendor_name.lower().strip()
    is_swapped = any(
        vendor_lower in bill_name or bill_name in vendor_lower
        for bill_name in bill_to_names
    )

    if is_swapped:
        logger.warning(f"Vendor/customer swap detected: '{vendor_name}' is actually customer")
        if not extracted_data.get("customer_name"):
            extracted_data["customer_name"] = vendor_name
        extracted_data["vendor_name"] = None

    return extracted_data

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
            # ── Fix vendor/customer swap ──
            result["extracted_data"] = fix_vendor_customer_swap(
                result["extracted_data"],
                raw_text
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