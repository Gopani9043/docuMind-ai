import os
import re
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
)

SQL_PROMPT = ChatPromptTemplate.from_template("""
You are an expert PostgreSQL analyst for a document processing system.
Generate an optimized PostgreSQL query for the question.

SCHEMA:
documents: id(UUID), filename(TEXT), status(TEXT: done|processing|error), created_at(TIMESTAMP)
extraction_results: id(UUID), doc_id(UUID→documents.id), document_type(TEXT: invoice|contract|receipt|report|unknown), extracted_data(JSON), raw_text(TEXT)

JSON FIELDS:
invoice: invoice_number, vendor_name, total_amount, currency, issue_date, due_date, vat_amount
contract: parties, start_date, end_date, value, currency, key_clauses
receipt: vendor_name, date, total_amount, currency
report: title, author, date, summary

RESOLVED CONTEXT:
{resolved_context}

CONVERSATION HISTORY:
{history}

CRITICAL NUMERIC RULE — ALWAYS use this pattern:
NULLIF(REPLACE(r.extracted_data->>'field', ',', ''), '')::numeric

NEVER do this:
(r.extracted_data->>'field')::numeric

DATE RULES:
today = CURRENT_DATE
this week = d.created_at >= DATE_TRUNC('week', NOW())
this month = d.created_at >= DATE_TRUNC('month', NOW())
last month = d.created_at >= DATE_TRUNC('month', NOW()) - INTERVAL '1 month' AND d.created_at < DATE_TRUNC('month', NOW())
last 30 days = d.created_at >= NOW() - INTERVAL '30 days'
overdue = TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY') < CURRENT_DATE

SAFETY RULES:
- Only SELECT queries
- Never DELETE/UPDATE/INSERT/DROP/ALTER/TRUNCATE
- Never access pg_catalog or information_schema
- Always JOIN: documents d JOIN extraction_results r ON r.doc_id = d.id
- Always WHERE d.status = 'done' unless querying errors
- LIMIT 200 for lists, 20 for aggregations

SYNONYMS:
invoice = bill|rechnung|faktura
contract = agreement|vertrag
receipt = kassenbon|payment
vendor = supplier|seller|company
amount = value|cost|price|total

EXAMPLE QUERIES:

-- All invoices
SELECT d.filename, r.extracted_data->>'vendor_name' as vendor, r.extracted_data->>'total_amount' as amount, r.extracted_data->>'currency' as currency, r.extracted_data->>'issue_date' as issue_date FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' ORDER BY d.created_at DESC LIMIT 200;

-- Top vendors by total paid
SELECT r.extracted_data->>'vendor_name' as vendor, COUNT(*) as invoice_count, ROUND(SUM(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric), 2) as total_paid, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND r.extracted_data->>'vendor_name' IS NOT NULL GROUP BY vendor, currency ORDER BY total_paid DESC LIMIT 20;

-- Document type distribution
SELECT r.document_type, COUNT(*) as count, ROUND(COUNT(*)*100.0/SUM(COUNT(*))OVER(), 1) as percentage FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE d.status = 'done' GROUP BY r.document_type ORDER BY count DESC;

-- Highest invoices
SELECT d.filename, r.extracted_data->>'vendor_name' as vendor, NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric as amount, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' ORDER BY amount DESC NULLS LAST LIMIT 10;

-- Average value by document type
SELECT r.document_type, ROUND(AVG(NULLIF(REPLACE(COALESCE(r.extracted_data->>'total_amount', r.extracted_data->>'value'),',',''),'')::numeric), 2) as avg_value, COUNT(*) as count FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE d.status = 'done' GROUP BY r.document_type HAVING AVG(NULLIF(REPLACE(COALESCE(r.extracted_data->>'total_amount', r.extracted_data->>'value'),',',''),'')::numeric) IS NOT NULL ORDER BY avg_value DESC;

-- Invoice closest to a specific value
SELECT d.filename, r.extracted_data->>'vendor_name' as vendor, NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric as amount, ABS(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric - {amount_reference}) as diff FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND r.extracted_data->>'total_amount' IS NOT NULL ORDER BY diff ASC LIMIT 5;

-- This month vs last month comparison
SELECT CASE WHEN d.created_at >= DATE_TRUNC('month', NOW()) THEN 'This Month' ELSE 'Last Month' END as period, COUNT(*) as count, ROUND(SUM(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric), 2) as total FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND d.created_at >= DATE_TRUNC('month', NOW()) - INTERVAL '1 month' GROUP BY period ORDER BY period;

-- Missing due dates
SELECT d.filename, r.extracted_data->>'vendor_name' as vendor, r.extracted_data->>'total_amount' as amount FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND (r.extracted_data->>'due_date' IS NULL OR r.extracted_data->>'due_date' = '') LIMIT 200;

-- Monthly totals trend
SELECT DATE_TRUNC('month', d.created_at) as month, COUNT(*) as count, ROUND(SUM(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric), 2) as total FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' GROUP BY month ORDER BY month DESC LIMIT 12;

-- Text search in documents
SELECT d.filename, r.document_type, r.extracted_data->>'vendor_name' as vendor FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE LOWER(r.raw_text) LIKE LOWER('%keyword%') AND d.status = 'done' LIMIT 200;

-- Invoices by currency with totals
SELECT r.extracted_data->>'currency' as currency, COUNT(*) as count, ROUND(SUM(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric), 2) as total FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' GROUP BY currency ORDER BY total DESC;

-- Failed documents
SELECT d.filename, d.status, d.created_at FROM documents d WHERE d.status = 'error' ORDER BY d.created_at DESC LIMIT 200;

-- Vendors appearing most frequently
SELECT r.extracted_data->>'vendor_name' as vendor, COUNT(*) as frequency, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE d.status = 'done' AND r.extracted_data->>'vendor_name' IS NOT NULL GROUP BY vendor, currency ORDER BY frequency DESC LIMIT 20;

-- EUR invoices above specific amount
SELECT d.filename, r.extracted_data->>'vendor_name' as vendor, NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric as amount FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND r.extracted_data->>'currency' = 'EUR' AND NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric > 5000 AND d.status = 'done' ORDER BY amount DESC LIMIT 200;

-- Processing status summary
SELECT status, COUNT(*) as count, ROUND(COUNT(*)*100.0/SUM(COUNT(*))OVER(), 1) as percentage FROM documents GROUP BY status ORDER BY count DESC;


Return ONLY the SQL query. No explanation. No markdown. No backticks.

Question: {question}
SQL:
""")

SAFE_KEYWORDS = {"SELECT"}
FORBIDDEN_KEYWORDS = {
    "DELETE", "UPDATE", "INSERT", "DROP",
    "ALTER", "TRUNCATE", "GRANT", "REVOKE"
}

def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL is safe to execute."""
    if not sql:
        return False, "Empty SQL"
    first_word = sql.strip().upper().split()[0]
    if first_word not in SAFE_KEYWORDS:
        return False, f"Unsafe operation: {first_word}"
    # Use word boundary regex — fixes false positive on created_at
    for forbidden in FORBIDDEN_KEYWORDS:
        if re.search(rf'\b{forbidden}\b', sql.upper()):
            return False, f"Forbidden keyword: {forbidden}"
    if any(t in sql.lower() for t in ["pg_catalog", "information_schema"]):
        return False, "System table access denied"
    return True, "OK"


def generate_sql(
    question: str,
    history: str = "",
    resolved_context: dict = None
) -> str:
    """Generate SQL from natural language question."""
    context = resolved_context or {}
    amount_ref = context.get("amount_reference", "0") or "0"

    try:
        chain = SQL_PROMPT | llm
        result = chain.invoke({
            "question": question,
            "history": history or "No history",
            "resolved_context": str(context),
            "amount_reference": amount_ref
        })

        sql = result.content.strip()

        # Clean markdown fences
        if "```" in sql:
            parts = sql.split("```")
            sql = parts[1] if len(parts) > 1 else parts[0]
            if sql.lower().startswith("sql"):
                sql = sql[3:]
        sql = sql.strip()

        logger.info(f"Generated SQL: {sql[:200]}")
        return sql

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return ""