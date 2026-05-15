import os
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

WICHTIG: Antworte NUR mit gültigem JSON. Kein Text davor oder danach.
Feldnamen IMMER auf Englisch.

Antworte in diesem Format:
{{"document_type": "invoice", "extracted_data": {{}}, "confidence": {{}}}}

DOKUMENT:
{text}
""")

def detect_language(text: str) -> str:
    """Detect language of text. Returns 'de' for German, 'en' for others."""
    try:
        lang = detect(text[:1000])
        logger.info(f"Detected language: {lang}")
        return lang
    except Exception:
        return "en"

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

        # If empty return default
        if not content:
            logger.warning("LLM returned empty content")
            return {"document_type": "unknown", "extracted_data": {}, "confidence": {}, "language": lang}

        result = json.loads(content)
        result["language"] = lang
        logger.info(f"Extraction successful: {result.get('document_type')} [{lang}]")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {"document_type": "unknown", "extracted_data": {}, "confidence": {}, "language": "unknown"}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise