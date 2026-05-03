import os
import json
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama3-70b-8192"),
    temperature=0,
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert document data extraction AI.
Analyze the document text below and extract structured information.

First, identify the document type: invoice, contract, receipt, report, or unknown.

Then extract ALL relevant fields based on the document type:

For INVOICE extract:
- invoice_number, issue_date, due_date, vendor_name,
  total_amount, currency, vat_amount, line_items (list)

For CONTRACT extract:
- parties (list of names), start_date, end_date,
  value, currency, key_clauses (list of important clauses)

For RECEIPT extract:
- vendor_name, date, total_amount, currency, items (list)

For REPORT extract:
- title, date, author, key_findings (list), summary

Return ONLY a valid JSON object with this exact structure:
{{
  "document_type": "invoice|contract|receipt|report|unknown",
  "extracted_data": {{ ...all extracted fields... }},
  "confidence": {{ "field_name": 0.0-1.0, ... }}
}}

Do not include any explanation or text outside the JSON.

DOCUMENT TEXT:
{text}
""")

def extract_structured_data(raw_text: str) -> dict:
    try:
        chain = EXTRACTION_PROMPT | llm
        response = chain.invoke({"text": raw_text[:8000]})

        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        logger.info(f"Extraction successful: {result.get('document_type')}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {
            "document_type": "unknown",
            "extracted_data": {},
            "confidence": {}
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise