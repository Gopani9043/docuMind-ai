import os
import re as _re
import re 
import logging
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from chatbot.llm_provider import invoke_with_fallback
load_dotenv()
logger = logging.getLogger(__name__)


SQL_PROMPT = ChatPromptTemplate.from_template(r"""
You are an expert PostgreSQL analyst for a document processing system.
Generate an optimized PostgreSQL query for the question.

SCHEMA:
documents: id(UUID), filename(TEXT), status(TEXT: done|processing|error), created_at(TIMESTAMP)
extraction_results: id(UUID), doc_id(UUID→documents.id), document_type(TEXT: invoice|contract|receipt|report|unknown), extracted_data(JSON), raw_text(TEXT)

JSON FIELDS:
invoice: invoice_number, vendor_name, customer_name, total_amount, currency, issue_date, due_date, vat_amount, line_items
contract: parties, start_date, end_date, value, currency, key_clauses
receipt: vendor_name, date, total_amount, currency
report: title, author, date, summary

RESOLVED CONTEXT:
{resolved_context}

CONVERSATION HISTORY:
{history}

CRITICAL NUMERIC RULE — ALWAYS use REGEXP_REPLACE to handle currency symbols:
CORRECT: NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount', '[^0-9.]', '', 'g'), '')::numeric
WRONG:   NULLIF(REPLACE(r.extracted_data->>'total_amount', ',', ''), '')::numeric
WRONG:   (r.extracted_data->>'total_amount')::numeric

REGEXP_REPLACE removes ALL non-numeric characters including £, €, $, ¥, commas, spaces.
Use this pattern for EVERY numeric cast in EVERY query.
DATE RULES — CRITICAL — use the DOCUMENT'S ACTUAL DATE (issue_date), not upload time (d.created_at):
- "this month", "last month", "this year", "last 30 days", "today",
  "newest", "oldest", "latest", "earliest" → ALL refer to issue_date
  (the date printed on the document), NOT d.created_at.
- ONLY use d.created_at when the question explicitly says "uploaded",
  "added to the system", "processed", "imported", or "scanned".

ALWAYS use this exact mixed-format parsing pattern for issue_date:

  (
    CASE
      WHEN r.extracted_data->>'issue_date' ~ '^\d{{4}}-\d{{2}}-\d{{2}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'YYYY-MM-DD')
      WHEN r.extracted_data->>'issue_date' ~ '^\d{{1,2}}/\d{{1,2}}/\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'DD/MM/YYYY')
      WHEN r.extracted_data->>'issue_date' ~ '^\d{{1,2}}\.\d{{1,2}}\.\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'DD.MM.YYYY')
      WHEN r.extracted_data->>'issue_date' ~ '^\d{{1,2}} \w+ \d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'DD Month YYYY')
      ELSE NULL
    END
  )

Write this CASE expression out IN FULL inline every time — never reference
it as if it were a plain column.

EXAMPLES (using PARSED_DATE to mean the full CASE expression above):
- today = PARSED_DATE = CURRENT_DATE
- this week = PARSED_DATE >= DATE_TRUNC('week', CURRENT_DATE)
- this month = PARSED_DATE >= DATE_TRUNC('month', CURRENT_DATE)
- last month = PARSED_DATE >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
               AND PARSED_DATE < DATE_TRUNC('month', CURRENT_DATE)
- this year = PARSED_DATE >= DATE_TRUNC('year', CURRENT_DATE)
- last 30 days = PARSED_DATE >= CURRENT_DATE - INTERVAL '30 days'

NEWEST/OLDEST RULE:
- "newest invoice/contract/receipt" / "latest" →
  ORDER BY PARSED_DATE DESC NULLS LAST LIMIT 1
- "oldest invoice/contract/receipt" / "earliest" →
  ORDER BY PARSED_DATE ASC NULLS LAST LIMIT 1
- For contracts, use start_date instead of issue_date in PARSED_DATE
  (same CASE pattern, different field).
- NEVER use DISTINCT ON for single-record queries (LIMIT 1) — just use
  ORDER BY + LIMIT.
- When LIMIT 1, never add DISTINCT ON — it is unnecessary and causes errors.
SAFETY RULES:
- Only SELECT queries
- Never DELETE/UPDATE/INSERT/DROP/ALTER/TRUNCATE
- Never access pg_catalog or information_schema
- Always JOIN: documents d JOIN extraction_results r ON r.doc_id = d.id
- Always WHERE d.status = 'done' unless querying errors
- LIMIT 200 for lists, 20 for aggregations
ORDINAL QUERIES:
- "second largest" → ORDER BY amount DESC NULLS LAST OFFSET 1 LIMIT 1
- "third largest"  → ORDER BY amount DESC NULLS LAST OFFSET 2 LIMIT 1
- "second smallest"→ ORDER BY amount ASC  NULLS LAST OFFSET 1 LIMIT 1
- "second newest"  → ORDER BY d.created_at DESC OFFSET 1 LIMIT 1
- "second oldest"  → ORDER BY d.created_at ASC  OFFSET 1 LIMIT 1
- Pattern: Nth item = OFFSET (N-1) LIMIT 1
- NEVER use DISTINCT ON for these — use a subquery if needed
OVERDUE + EXPIRING CROSS-DOC PATTERN:
"overdue invoices from vendors who also have expiring contracts" →
Primary result: overdue INVOICES (due_date < CURRENT_DATE)
Filter: vendor must appear in a contract expiring within 30 days

PREVIOUS QUERY (if this is a follow-up, use its filters as context):
{previous_sql}

If the user says "only X" or "just X" or filters by vendor/currency/amount,
apply that filter ON TOP OF the previous query's filters (currency, amount threshold, doc type).

SELECT DISTINCT ON (d.filename) d.filename,
    r.extracted_data->>'vendor_name' as vendor,
    r.extracted_data->>'due_date' as due_date,
    r.extracted_data->>'total_amount' as amount,
    r.extracted_data->>'currency' as currency
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'invoice'
  AND d.status = 'done'
  AND r.extracted_data->>'due_date' IS NOT NULL
  AND (
    CASE
        WHEN r.extracted_data->>'due_date' ~ '^\d{{4}}-\d{{2}}-\d{{2}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'YYYY-MM-DD')
        WHEN r.extracted_data->>'due_date' ~ '^\d{{1,2}}/\d{{1,2}}/\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD/MM/YYYY')
        WHEN r.extracted_data->>'due_date' ~ '^\d{{1,2}}\.\d{{1,2}}\.\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD.MM.YYYY')
        WHEN r.extracted_data->>'due_date' ~ '^\d{{1,2}} \w+ \d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY')
        ELSE NULL
    END
  ) < CURRENT_DATE
  AND EXISTS (
    SELECT 1 FROM documents d2
    JOIN extraction_results r2 ON r2.doc_id = d2.id
    WHERE r2.document_type = 'contract'
      AND d2.status = 'done'
      AND r2.extracted_data->>'parties' ILIKE CONCAT('%', r.extracted_data->>'vendor_name', '%')
      AND r2.extracted_data->>'end_date' IS NOT NULL
      AND (
        CASE
            WHEN r2.extracted_data->>'end_date' ~ '^\d{{4}}-\d{{2}}-\d{{2}}$'
            THEN TO_DATE(r2.extracted_data->>'end_date', 'YYYY-MM-DD')
            WHEN r2.extracted_data->>'end_date' ~ '^\d{{1,2}}/\d{{1,2}}/\d{{4}}$'
            THEN TO_DATE(r2.extracted_data->>'end_date', 'DD/MM/YYYY')
            WHEN r2.extracted_data->>'end_date' ~ '^\d{{1,2}}\.\d{{1,2}}\.\d{{4}}$'
            THEN TO_DATE(r2.extracted_data->>'end_date', 'DD.MM.YYYY')
            WHEN r2.extracted_data->>'end_date' ~ '^\d{{1,2}} \w+ \d{{4}}$'
            THEN TO_DATE(r2.extracted_data->>'end_date', 'DD Month YYYY')
            ELSE NULL
        END
      ) BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
  )
ORDER BY d.filename LIMIT 200;
COMBINED FILTER + DUPLICATE PATTERN:
"Show [filtered] invoices, find duplicates, show [oldest/newest]" →
Generate ONE SQL that returns ALL filtered invoices WITH a has_duplicate flag.
NEVER filter out non-duplicates — always show all results with the flag.

Example — "EUR invoices above 10000, find duplicates, show oldest":
SELECT DISTINCT ON (d.filename) d.filename,
    r.extracted_data->>'vendor_name' as vendor,
    r.extracted_data->>'invoice_number' as invoice_number,
    r.extracted_data->>'total_amount' as amount,
    r.extracted_data->>'currency' as currency,
    r.extracted_data->>'issue_date' as issue_date,
    CASE WHEN r.extracted_data->>'invoice_number' IN (
        SELECT r2.extracted_data->>'invoice_number'
        FROM extraction_results r2
        WHERE r2.document_type = 'invoice'
          AND r2.extracted_data->>'invoice_number' IS NOT NULL
        GROUP BY r2.extracted_data->>'invoice_number'
        HAVING COUNT(*) > 1
    ) THEN 'Yes' ELSE 'No' END as has_duplicate
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'invoice'
  AND d.status = 'done'
  AND r.extracted_data->>'currency' = 'EUR'
  AND NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric > 10000
ORDER BY d.filename, d.created_at ASC
LIMIT 200;

This returns ALL 9 invoices. The has_duplicate column tells which ones are repeated.
"oldest" → ORDER BY d.created_at ASC (first row is oldest)
"newest" → ORDER BY d.created_at DESC

Rules:
- Filter goes in outer WHERE clause
- Duplicate detection uses IN (subquery with HAVING COUNT(*) > 1)
- "oldest" → ORDER BY d.created_at ASC
- "newest" → ORDER BY d.created_at DESC
- Currency filter: AND r.extracted_data->>'currency' = 'EUR'
- Amount filter: AND NULLIF(REGEXP_REPLACE(...,'[^0-9.]','','g'),'')::numeric > X

DEDUPLICATION RULE (ALWAYS apply):
- Use DISTINCT ON (d.filename) for list queries
- Always ORDER BY d.filename, d.created_at DESC with DISTINCT ON
- Never return same filename twice
CONTRACT VENDOR RULE — CRITICAL:
- Contracts do NOT have vendor_name field
- Contracts store parties as JSON array in extracted_data->>'parties'
- To search contracts by vendor use: r.extracted_data->>'parties' ILIKE '%vendor_name%'
- Never use extracted_data->>'vendor_name' for contracts
- For invoices/receipts use: extracted_data->>'vendor_name'
- For contracts use: extracted_data->>'parties' ILIKE '%name%'

MONTHLY TREND RULE — CRITICAL (universal for any vendor/currency):
- "monthly trend", "trend over time", "trend for [vendor]" →
  ALWAYS a single GROUP BY query using issue_date (never d.created_at)
- Use this exact mixed-format date parsing pattern:

  DATE_TRUNC('month',
    CASE
      WHEN r.extracted_data->>'issue_date' ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'YYYY-MM-DD')
      WHEN r.extracted_data->>'issue_date' ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'DD/MM/YYYY')
      WHEN r.extracted_data->>'issue_date' ~ '^\\d{{1,2}} \\w+ \\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'DD Month YYYY')
      ELSE NULL
    END
  ) AS month

- ALWAYS select BOTH: COUNT(*) as invoice_count AND ROUND(SUM(amount),2) as total_amount
- ALWAYS add: AND r.extracted_data->>'issue_date' IS NOT NULL (prevents NULL month rows)
- If a vendor is mentioned: AND r.extracted_data->>'vendor_name' ILIKE '%[vendor]%'
- GROUP BY month ORDER BY month ASC (chronological order, oldest first)
- Example:
  SELECT
    DATE_TRUNC('month', CASE
      WHEN r.extracted_data->>'issue_date' ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
        THEN TO_DATE(r.extracted_data->>'issue_date', 'YYYY-MM-DD')
      ELSE NULL
    END) as month,
    COUNT(*) as invoice_count,
    ROUND(SUM(NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric), 2) as total_amount
  FROM documents d JOIN extraction_results r ON r.doc_id = d.id
  WHERE r.document_type = 'invoice' AND d.status = 'done'
  AND r.extracted_data->>'vendor_name' ILIKE '%BrightPath Analytics%'
  AND r.extracted_data->>'issue_date' IS NOT NULL
  GROUP BY month ORDER BY month ASC;

ORDINAL RANKING RULE — works for ANY ordinal, ANY metric, ANY language:
"Nth highest/most/largest X" → ORDER BY metric DESC LIMIT 1 OFFSET (N-1)
"Nth lowest/least/smallest X" → ORDER BY metric ASC LIMIT 1 OFFSET (N-1)

Example — "second most invoices per vendor":
SELECT r.extracted_data->>'vendor_name' as vendor, COUNT(*) as cnt
FROM documents d JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'invoice' AND d.status = 'done'
GROUP BY vendor ORDER BY cnt DESC LIMIT 1 OFFSET 1;

Example — "third highest spending vendor":
SELECT r.extracted_data->>'vendor_name' as vendor, SUM(NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric) as total
FROM documents d JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'invoice' AND d.status = 'done'
GROUP BY vendor ORDER BY total DESC LIMIT 1 OFFSET 2;

This is a general rule — never hardcode specific ordinal words; compute OFFSET as (position - 1) for whatever ordinal the user asks for.

VENDOR TOTAL RULE:
- When asking "which vendor paid most" or "top vendor" — GROUP BY vendor only, NOT by currency
- Never GROUP BY vendor AND currency together for top vendor queries
- Currency grouping only when user explicitly asks "by currency" or "in EUR"

DOCUMENT TYPE RULE — CRITICAL:
- "what document types" or "which types" or "document type breakdown" →
  MUST use GROUP BY r.document_type — NEVER use DISTINCT ON
  CORRECT SQL: SELECT r.document_type, COUNT(*) as count FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE d.status = 'done' GROUP BY r.document_type ORDER BY count DESC;
- "show receipts" → WHERE r.document_type = 'receipt'
- "show contracts" → WHERE r.document_type = 'contract'
- "show invoices" → WHERE r.document_type = 'invoice'
- NEVER mix document types unless user asks for "all documents"
VENDOR SEARCH RULE:
- Always use ILIKE '%name%' not = 'name' for vendor searches
- This catches BrightPath Analytics AND BrightPath Analytics Ltd together
- Example: LOWER(r.extracted_data->>'vendor_name') LIKE LOWER('%BrightPath%')
CUSTOMER NAME RULE:
- vendor_name = company that issued the invoice (seller)
- customer_name = company that received the invoice (buyer)
- "invoices sent to Acme" → WHERE extracted_data->>'customer_name' ILIKE '%Acme%'
- "invoices from BrightPath" → WHERE extracted_data->>'vendor_name' ILIKE '%BrightPath%'
- "what did we buy from X" → use vendor_name
- "who did we sell to" → use customer_name
OVERDUE DATE RULE — CRITICAL:
Dates are stored in mixed formats: YYYY-MM-DD, DD/MM/YYYY, DD Month YYYY
Use this safe pattern that handles all formats:

-- Most expensive / high-value contracts
SELECT DISTINCT ON (d.filename) d.filename,
       r.extracted_data->>'parties' as parties,
       NULLIF(REGEXP_REPLACE(r.extracted_data->>'value','[^0-9.]','','g'),'')::numeric as value,
       r.extracted_data->>'currency' as currency
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'contract' AND d.status = 'done'
AND r.extracted_data->>'value' IS NOT NULL
ORDER BY d.filename, value DESC
LIMIT 10;

TOTAL AMOUNT RULE — CRITICAL:
- ANY question about total spending/money/payments without a specific currency
  ("how much did we spend", "how much money", "what did we spend",
  "total invoices", "how much total", "total spending", "how much have we paid") 
  NEVER sum across currencies — always GROUP BY currency
  SELECT r.extracted_data->>'currency' as currency,
         ROUND(SUM(NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric), 2) as total_amount
  FROM documents d JOIN extraction_results r ON r.doc_id = d.id
  WHERE r.document_type = 'invoice' AND d.status = 'done'
  GROUP BY currency ORDER BY total_amount DESC;
- Only sum without currency grouping when user explicitly asks for a specific currency:
  "total EUR invoices" → WHERE currency = 'EUR' then SUM

WHERE r.extracted_data->>'due_date' IS NOT NULL
AND r.extracted_data->>'due_date' != ''
AND (
    CASE
        WHEN r.extracted_data->>'due_date' ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'YYYY-MM-DD')
        WHEN r.extracted_data->>'due_date' ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD/MM/YYYY')
        WHEN r.extracted_data->>'due_date' ~ '^\\d{{1,2}} \\w+ \\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY')
        WHEN r.extracted_data->>'due_date' ~ '^\\d{{1,2}}\.\\d{{1,2}}\.\\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD.MM.YYYY')
        ELSE NULL
    END
) < CURRENT_DATE

EXPIRING CONTRACT RULE — same pattern for end_date:
WHERE r.extracted_data->>'end_date' IS NOT NULL
AND r.extracted_data->>'end_date' != ''
AND (
    CASE
        WHEN r.extracted_data->>'end_date' ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
        THEN TO_DATE(r.extracted_data->>'end_date', 'YYYY-MM-DD')
        WHEN r.extracted_data->>'end_date' ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'end_date', 'DD/MM/YYYY')
        WHEN r.extracted_data->>'end_date' ~ '^\\d{{1,2}} \\w+ \\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'end_date', 'DD Month YYYY')
        WHEN r.extracted_data->>'due_date' ~ '^\\d{{1,2}}\.\\d{{1,2}}\.\\d{{4}}$'
        THEN TO_DATE(r.extracted_data->>'due_date', 'DD.MM.YYYY')
        ELSE NULL
    END
) BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
GROWTH / CHANGE BETWEEN PERIODS PATTERN (issue_date based — for "which vendor/currency grew most"):
- Use issue_date (not created_at) for "grew most between months" style questions —
  created_at is upload time and not meaningful for trend analysis.
- Compare each group's value to its PREVIOUS month-with-data (LAG), not strictly
  the prior calendar month — sparse data means consecutive calendar months may not exist.
- ALWAYS include both the current and previous period in the output (as 'month' and
  'prev_month') so the comparison period is visible to the user.
- Pattern (replace 'vendor_name' with currency/document_type/etc as needed):

GROWTH KEYWORDS:
grew most|grew the most|biggest increase|biggest growth|increased the most|
biggest decline|dropped the most|decreased the most|biggest change between months|
declined most|declined the most|biggest drop|fell the most|decreased most
→ use the GROWTH / CHANGE BETWEEN PERIODS pattern above
- For "grew/increased/biggest growth" → ORDER BY growth_percent DESC
- For "declined/dropped/decreased/fell" → ORDER BY growth_percent ASC

WITH monthly AS (
  SELECT
    r.extracted_data->>'vendor_name' AS vendor,
    DATE_TRUNC('month',
      CASE
        WHEN r.extracted_data->>'issue_date' ~ '^\d{{4}}-\d{{2}}-\d{{2}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'YYYY-MM-DD')
        WHEN r.extracted_data->>'issue_date' ~ '^\d{{1,2}}/\d{{1,2}}/\d{{4}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'DD/MM/YYYY')
        WHEN r.extracted_data->>'issue_date' ~ '^\d{{1,2}} \w+ \d{{4}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'DD Month YYYY')
        WHEN r.extracted_data->>'issue_date' ~ '^\d{{1,2}}\.\d{{1,2}}\.\d{{4}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'DD.MM.YYYY')
        ELSE NULL
      END
    ) AS month,
    NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount', '[^0-9.]', '', 'g'), '')::numeric AS amount
  FROM documents d
  JOIN extraction_results r ON r.doc_id = d.id
  WHERE r.document_type = 'invoice'
    AND d.status = 'done'
    AND r.extracted_data->>'vendor_name' IS NOT NULL
),
totals AS (
  SELECT vendor, month, ROUND(SUM(amount), 2) AS total
  FROM monthly
  WHERE month IS NOT NULL
  GROUP BY vendor, month
),
growth AS (
  SELECT
    vendor, month, total,
    LAG(total) OVER (PARTITION BY vendor ORDER BY month) AS prev_total,
    LAG(month) OVER (PARTITION BY vendor ORDER BY month) AS prev_month
  FROM totals
)
SELECT vendor, prev_month, month, prev_total, total,
       ROUND((total - prev_total) / NULLIF(prev_total, 0) * 100, 2) AS growth_percent
FROM growth
WHERE prev_total IS NOT NULL
ORDER BY growth_percent DESC
LIMIT 1;

SYNONYMS:
invoice = bill|rechnung|faktura
contract = agreement|vertrag
receipt = kassenbon|payment
vendor = supplier|seller|company|lieferant
amount = value|cost|price|total

EXAMPLE QUERIES:

-- All invoices (DISTINCT to prevent duplicates)
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'vendor_name' as vendor, r.extracted_data->>'total_amount' as amount, r.extracted_data->>'currency' as currency, r.extracted_data->>'issue_date' as issue_date FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' ORDER BY d.filename, d.created_at DESC LIMIT 200;

-- Top vendors by total paid across all currencies
SELECT r.extracted_data->>'vendor_name' as vendor, COUNT(*) as invoice_count, ROUND(SUM(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric), 2) as total_paid FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND r.extracted_data->>'vendor_name' IS NOT NULL GROUP BY vendor ORDER BY total_paid DESC LIMIT 20;

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

-- All receipts
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'vendor_name' as vendor, r.extracted_data->>'total_amount' as amount, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'receipt' AND d.status = 'done' ORDER BY d.filename LIMIT 200;

-- EUR invoices only
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'vendor_name' as vendor, NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric as amount, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND r.extracted_data->>'currency' = 'EUR' ORDER BY d.filename LIMIT 200;

-- Invoices from vendor using partial match
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'vendor_name' as vendor, NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric as amount, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND LOWER(r.extracted_data->>'vendor_name') LIKE LOWER('%BrightPath%') ORDER BY amount DESC NULLS LAST LIMIT 200;

-- Overdue invoices (handles mixed date formats)
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'vendor_name' as vendor, r.extracted_data->>'due_date' as due_date, r.extracted_data->>'total_amount' as amount FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'invoice' AND d.status = 'done' AND r.extracted_data->>'due_date' IS NOT NULL AND r.extracted_data->>'due_date' != '' AND (CASE WHEN r.extracted_data->>'due_date' ~ '^\d{{4}}-\d{{2}}-\d{{2}}$' THEN TO_DATE(r.extracted_data->>'due_date', 'YYYY-MM-DD') WHEN r.extracted_data->>'due_date' ~ '^\d{{1,2}}/\d{{1,2}}/\d{{4}}$' THEN TO_DATE(r.extracted_data->>'due_date', 'DD/MM/YYYY') WHEN r.extracted_data->>'due_date' ~ '^\d{{1,2}} \w+ \d{{4}}$' THEN TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY') ELSE NULL END) < CURRENT_DATE ORDER BY d.filename LIMIT 200;

-- Contracts expiring soon (handles mixed date formats)
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'parties' as parties, r.extracted_data->>'end_date' as end_date, r.extracted_data->>'value' as value FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'contract' AND d.status = 'done' AND r.extracted_data->>'end_date' IS NOT NULL AND r.extracted_data->>'end_date' != '' AND (CASE WHEN r.extracted_data->>'end_date' ~ '^\d{{4}}-\d{{2}}-\d{{2}}$' THEN TO_DATE(r.extracted_data->>'end_date', 'YYYY-MM-DD') WHEN r.extracted_data->>'end_date' ~ '^\d{{1,2}}/\d{{1,2}}/\d{{4}}$' THEN TO_DATE(r.extracted_data->>'end_date', 'DD/MM/YYYY') WHEN r.extracted_data->>'end_date' ~ '^\d{{1,2}} \w+ \d{{4}}$' THEN TO_DATE(r.extracted_data->>'end_date', 'DD Month YYYY') ELSE NULL END) BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days' ORDER BY d.filename LIMIT 200;

-- All contracts
SELECT DISTINCT ON (d.filename) d.filename, r.extracted_data->>'parties' as parties, r.extracted_data->>'value' as value, r.extracted_data->>'currency' as currency FROM documents d JOIN extraction_results r ON r.doc_id = d.id WHERE r.document_type = 'contract' AND d.status = 'done' ORDER BY d.filename LIMIT 200;

-- Largest single invoice from a specific vendor
SELECT d.filename,
       r.extracted_data->>'vendor_name' AS vendor,
       NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric AS amount,
       r.extracted_data->>'currency' AS currency,
       r.extracted_data->>'issue_date' AS issue_date
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'invoice'
  AND d.status = 'done'
  AND LOWER(r.extracted_data->>'vendor_name') = LOWER('BrightPath Analytics')
ORDER BY amount DESC NULLS LAST
LIMIT 1;

-- Contracts from a specific vendor (use parties array, not vendor_name)
SELECT d.filename,
       r.extracted_data->>'parties' as parties,
       r.extracted_data->>'value' as value,
       r.extracted_data->>'currency' as currency,
       r.extracted_data->>'start_date' as start_date,
       r.extracted_data->>'end_date' as end_date
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'contract'
  AND d.status = 'done'
  AND r.extracted_data->>'parties' ILIKE '%BrightPath Analytics%'
ORDER BY d.created_at DESC
LIMIT 200;

-- Top vendor by total paid across ALL currencies
SELECT r.extracted_data->>'vendor_name' as vendor,
       ROUND(SUM(NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric), 2) as total_paid
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = 'invoice'
  AND d.status = 'done'
  AND r.extracted_data->>'vendor_name' IS NOT NULL
GROUP BY vendor
ORDER BY total_paid DESC
LIMIT 1;
UPLOAD DATE RULE:
- "when was it uploaded" or "upload date" → query d.created_at
- Always include d.filename and vendor in result
- Use vendor from context to filter: WHERE LOWER(vendor_name) = LOWER('[vendor]')
- Example: SELECT d.filename, r.extracted_data->>'vendor_name' as vendor, 
           d.created_at as uploaded_at
           FROM documents d JOIN extraction_results r ON r.doc_id = d.id
           WHERE d.status = 'done'
           AND LOWER(r.extracted_data->>'vendor_name') = LOWER('[vendor]')
           ORDER BY d.created_at DESC LIMIT 1
Return ONLY the SQL query. No explanation. No markdown. No backticks.
Question: {question}
SQL:
""")

SAFE_KEYWORDS = {"SELECT", "WITH"}
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

def fix_distinct_order_conflict(sql: str) -> str:
    """
    PostgreSQL requires DISTINCT ON (x) to have x as the first ORDER BY column.
    Solution: wrap in subquery, outer ORDER BY only uses columns in SELECT list.
    """
    sql_upper = sql.upper()

    # ── Guard: don't wrap if SQL has complex nested IN/EXISTS subqueries ──────
    # Wrapping causes "syntax error at or near _sub" when inner query has
    # unclosed nested subqueries (e.g. IN (SELECT ... EXISTS (...)))
    if "IN (" in sql_upper and "EXISTS (" in sql_upper:
        depth = 0
        for ch in sql:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
        if depth != 0:
            logger.info("fix_distinct: skipping wrap — unbalanced parentheses in nested subquery")
            return sql

    if "DISTINCT ON" not in sql_upper:
        return sql

    # LIMIT 1 — remove DISTINCT ON entirely, no dedup needed
    if re.search(r'\bLIMIT\s+1\b', sql_upper):
        sql = re.sub(r'DISTINCT ON\s*\([^)]+\)\s*', '', sql, flags=re.IGNORECASE)
        logger.info("fix_distinct: removed DISTINCT ON for LIMIT 1 query")
        return sql

    order_pos = sql_upper.rfind("ORDER BY")
    if order_pos == -1:
        return sql

    order_clause = sql[order_pos:]
    inner_sql = sql[:order_pos].strip()

    # Extract LIMIT
    limit_match = re.search(r'LIMIT\s+\d+', order_clause, re.IGNORECASE)
    limit_clause = limit_match.group(0) if limit_match else "LIMIT 200"

    # Extract the sort direction (ASC or DESC) from the original ORDER BY
    direction = "DESC"
    if re.search(r'\bASC\b', order_clause, re.IGNORECASE):
        direction = "ASC"

    # Detect what field to sort by in outer query — only use columns in SELECT
    outer_sort = "filename"  # safe default — always in SELECT
    if re.search(r'\bamount\b', order_clause, re.IGNORECASE):
        outer_sort = "amount"
    elif re.search(r'\bissue_date\b', order_clause, re.IGNORECASE):
        outer_sort = "issue_date"
    elif re.search(r'\bdue_date\b', order_clause, re.IGNORECASE):
        outer_sort = "due_date"
    elif re.search(r'\bend_date\b', order_clause, re.IGNORECASE):
        outer_sort = "end_date"
    elif re.search(r'\bstart_date\b', order_clause, re.IGNORECASE):
        outer_sort = "start_date"
    elif re.search(r'\bvalue\b', order_clause, re.IGNORECASE):
        outer_sort = "value"
    elif re.search(r'\bcreated_at\b', order_clause, re.IGNORECASE):
        # created_at is NOT in SELECT — use filename instead
        outer_sort = "filename"

    rewritten = (
        f"SELECT * FROM (\n"
        f"  {inner_sql}\n"
        f"  ORDER BY d.filename, d.created_at DESC\n"
        f") _sub\n"
        f"ORDER BY {outer_sort} {direction} NULLS LAST {limit_clause}"
    )

    logger.info(f"fix_distinct: wrapped in subquery, outer sort={outer_sort} {direction}")
    return rewritten

def fix_spurious_duplicate_check(sql: str, question: str) -> str:
    """
    Remove has_duplicate CASE expression when user didn't ask about duplicates.
    The LLM sometimes adds this check speculatively, causing all NULL-invoice-number
    rows to be incorrectly flagged as duplicates.
    """
    duplicate_intent_words = [
        "duplicate", "repeat", "same invoice", "appears twice",
        "invoice number", "repeated"
    ]
    if any(w in question.lower() for w in duplicate_intent_words):
        return sql  # user asked about duplicates — keep the check

    # Remove the entire has_duplicate CASE expression
    sql = re.sub(
        r",?\s*CASE\s+WHEN\s+r\d*\.extracted_data->>'invoice_number'\s+IN\s*"
        r"\(\s*SELECT[^)]+HAVING COUNT\(\*\)\s*>\s*1\s*\)\s*"
        r"THEN\s+'Yes'\s+ELSE\s+'No'\s+END\s+as\s+has_duplicate",
        "",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )
    if "has_duplicate" not in sql:
        logger.info("fix_spurious_duplicate_check: removed unnecessary duplicate check")
    return sql

def fix_duplicate_null_check(sql: str) -> str:
    """
    Fix duplicate invoice number detection subquery to exclude NULLs.
    Without this, all invoices with NULL invoice_number are flagged as duplicates.
    """
    # Pattern: HAVING COUNT(*) > 1 inside a duplicate-check subquery
    # Add AND invoice_number IS NOT NULL AND invoice_number != '' to the WHERE clause
    sql = re.sub(
        r"(WHERE\s+r\d*\.document_type\s*=\s*'invoice'\s*\n?\s*"
        r"AND\s+r\d*\.extracted_data->>'invoice_number'\s+IS\s+NOT\s+NULL)"
        r"(\s*\n?\s*GROUP BY\s+r\d*\.extracted_data->>'invoice_number'\s*\n?\s*HAVING COUNT\(\*\)\s*>\s*1)",
        r"\1\n    AND r2.extracted_data->>'invoice_number' != ''"
        r"\2",
        sql,
        flags=re.IGNORECASE
    )
    if "HAVING COUNT(*) > 1" in sql:
        logger.info("fix_duplicate_null_check: applied")
    return sql

def fix_vendor_exact_match(sql: str) -> str:
    """
    Convert exact vendor matches to ILIKE partial matches.
    LOWER(field) = LOWER('name') → field ILIKE '%name%'
    This is universal — works for any vendor name.
    """
    # Pattern: LOWER(r.extracted_data->>'vendor_name') = LOWER('SomeName')
    pattern = r"LOWER\(r\.extracted_data->>'vendor_name'\)\s*=\s*LOWER\('([^']+)'\)"
    
    def replace_with_ilike(match):
        vendor_name = match.group(1)
        return f"r.extracted_data->>'vendor_name' ILIKE '%{vendor_name}%'"
    
    fixed = re.sub(pattern, replace_with_ilike, sql, flags=re.IGNORECASE)
    
    if fixed != sql:
        logger.info(f"fix_vendor: converted exact match to ILIKE for vendor")
    
    return fixed

def fix_vendor_canonical_names(sql: str, all_vendors: list) -> str:
    """
    After SQL is generated, find any vendor name strings in IN(...) or ILIKE clauses
    and replace them with their canonical DB names via fuzzy matching.
    Universal — works for any vendor, any SQL pattern the LLM generates.
    """
    if not all_vendors or not sql:
        return sql

    from services.vendor_matcher import find_matches, similarity
    CURRENCY_CODES = {
        "EUR", "USD", "GBP", "JPY", "CHF", "INR", 
        "CAD", "AUD", "CNY", "SEK", "NOK", "DKK"
    }

    # ── Never replace PDF filenames — they are document identifiers, not
    # vendor names. Protect all 'something.pdf' strings from being treated
    # as vendor candidates before any fuzzy matching runs. ──
    pdf_pattern = re.compile(r"'[^']*\.pdf'", re.IGNORECASE)
    protected = {}
    placeholder_idx = 0

    def protect_filename(match):
        nonlocal placeholder_idx
        key = f"__PDF_PLACEHOLDER_{placeholder_idx}__"
        protected[key] = match.group(0)
        placeholder_idx += 1
        return f"'{key}'"

    sql = pdf_pattern.sub(protect_filename, sql)

    def replace_vendor(match_str):
        # Extract the quoted string value
        quoted = re.findall(r"'([^']+)'", match_str)
        if not quoted:
            return match_str
        
        result = match_str
        for name in quoted:
            if len(name) < 3:
                continue
            # ── Never replace currency codes ──
            if name.upper() in CURRENCY_CODES:
                continue
            # Skip if already an exact canonical match
            if name in all_vendors:
                continue
            # Find best canonical match
            fuzzy = find_matches(name, all_vendors, threshold=0.55)
            substring = [v for v in all_vendors if name.lower() in v.lower()]
            candidates = list({*fuzzy, *substring})
            if candidates:
                best = max(candidates, key=lambda v: similarity(name, v))
                if best.lower() != name.lower():
                    logger.info(f"fix_vendor_canonical: '{name}' → '{best}'")
                    result = result.replace(f"'{name}'", f"'{best}'")
        return result

    # Fix IN (...) clauses — e.g. IN (LOWER('BrightPath'), LOWER('FinEdge'))
    sql = re.sub(
        r'IN\s*\([^)]+\)',
        lambda m: replace_vendor(m.group(0)),
        sql,
        flags=re.IGNORECASE
    )

    # Fix ILIKE '%name%' clauses — preserve the % wildcards
    def replace_ilike_vendor(match):
        full = match.group(0)
        inner = re.search(r"'%([^']+)%'", full)
        if not inner:
            return full
        name = inner.group(1)
        if name in all_vendors:
            return full
        fuzzy = find_matches(name, all_vendors, threshold=0.55)
        substring = [v for v in all_vendors if name.lower() in v.lower()]
        candidates = list({*fuzzy, *substring})
        if candidates:
            best = max(candidates, key=lambda v: similarity(name, v))
            if best.lower() != name.lower():
                logger.info(f"fix_vendor_canonical: ILIKE '%{name}%' → '%{best}%'")
                return f"ILIKE '%{best}%'"
        return full

    sql = re.sub(
        r"ILIKE\s*'%[^']+%'",
        replace_ilike_vendor,
        sql,
        flags=re.IGNORECASE
    )

    # Fix = 'name' exact match clauses (vendor_name = 'BrightPath')
    sql = re.sub(
        r"=\s*'([^']+)'",
        lambda m: replace_vendor(m.group(0)),
        sql,
        flags=re.IGNORECASE
    )

    # ── Restore protected filenames ──
    for key, original in protected.items():
        sql = sql.replace(f"'{key}'", original)

    return sql

def fix_missing_currency_filter(sql: str, question: str) -> str:
    """
    If user mentions a currency with an amount (e.g. 'above 10000 EUR'),
    ensure the SQL includes a currency filter.
    Works for any currency — universal.
    """
    # Detect currency mentioned in question
    currency_pattern = r'\b(EUR|USD|GBP|JPY|CHF|INR|CAD|AUD|CNY|SEK|NOK|DKK)\b'
    currencies_in_question = re.findall(currency_pattern, question.upper())
    
    if not currencies_in_question:
        return sql  # no currency mentioned — nothing to fix
    
    currency = currencies_in_question[0]  # take first mentioned currency
    
    # Check if SQL already has a currency filter
    if f"'{currency}'" in sql or f'"{currency}"' in sql:
        return sql  # already filtered — nothing to do
    
    # Only inject for amount-based queries
    amount_keywords = ['> ', '< ', '>= ', '<= ', 'above', 'below', 
                       'more than', 'less than', 'over', 'under']
    has_amount_filter = any(k in question.lower() for k in amount_keywords)
    
    if not has_amount_filter:
        return sql  # no amount filter — currency filter not needed
    
    # Inject currency filter before ORDER BY or LIMIT
    currency_filter = f"AND r.extracted_data->>'currency' = '{currency}'"
    
    sql_upper = sql.upper()
    
    # Find injection point — before ORDER BY, LIMIT, or end of WHERE clause
    inject_pos = None
    for keyword in ['ORDER BY', 'LIMIT', ') _SUB']:
        pos = sql_upper.find(keyword)
        if pos != -1:
            inject_pos = pos
            break
    
    if inject_pos:
        sql = sql[:inject_pos] + f"{currency_filter}\n  " + sql[inject_pos:]
        logger.info(f"fix_currency: injected {currency} filter")
    
    return sql

def fix_nullif_syntax(sql: str) -> str:
    """
    Fix malformed NULLIF where LLM forgets the second argument.
    NULLIF(expr::numeric) → NULLIF(expr, '')::numeric
    Also fixes NULLIF(REGEXP_REPLACE(...)::numeric) pattern.
    """
    # Pattern: NULLIF(REGEXP_REPLACE(..., '[^0-9.]', '', 'g')::numeric)
    # Should be: NULLIF(REGEXP_REPLACE(..., '[^0-9.]', '', 'g'), '')::numeric
    broken = re.compile(
        r"NULLIF\(("
        r"REGEXP_REPLACE\([^)]+\)"
        r")::numeric\)",
        re.IGNORECASE
    )
    fixed = broken.sub(r"NULLIF(\1, '')::numeric", sql)
    if fixed != sql:
        logger.info("fix_nullif: corrected malformed NULLIF syntax")
    return fixed

def fix_outer_subquery_references(sql: str) -> str:
    """
    When SQL has the form SELECT ... FROM ( SELECT ... ) alias ...,
    the OUTER SELECT must reference the subquery's OUTPUT column names
    directly (e.g. 'vendor', 'this_month_total'), not re-derive them
    from the original tables (r.field / d.field) — those tables don't
    exist in the outer scope. This rewrites such re-derivations down
    to plain column references. Universal — works for any field/alias.
    """
    match = re.search(r'\bFROM\s*\(', sql, re.IGNORECASE)
    if not match or not sql.strip().upper().startswith("SELECT"):
        return sql

    outer_select = sql[:match.start()]
    rest = sql[match.start():]
    original = outer_select

    # r.extracted_data->>'field' as alias  ->  alias
    outer_select = re.sub(
        r"\b\w+\.extracted_data->>'(\w+)'\s+as\s+(\w+)",
        r"\2",
        outer_select,
        flags=re.IGNORECASE
    )
    # r.extracted_data->>'field' (no alias)  ->  field
    outer_select = re.sub(
        r"\b\w+\.extracted_data->>'(\w+)'",
        r"\1",
        outer_select,
        flags=re.IGNORECASE
    )
    # d.column as alias  ->  alias   (e.g. d.created_at as uploaded_at)
    outer_select = re.sub(
        r"\b[a-zA-Z_][\w]*\.(\w+)\s+as\s+(\w+)",
        r"\2",
        outer_select,
        flags=re.IGNORECASE
    )
    # leftover bare table.column  ->  column   (e.g. r.vendor, d.created_at)
    outer_select = re.sub(
        r"\b[a-zA-Z_][\w]*\.(\w+)\b",
        r"\1",
        outer_select
    )

    if outer_select != original:
        logger.info("fix_outer_subquery: rewrote outer SELECT to reference subquery columns")

    return outer_select + rest

def fix_subquery_to_cte(sql: str) -> str:
    """
    When the LLM generates nested subqueries with JOIN that reference
    earlier subquery aliases (e.g. FROM (SELECT ...) AS monthly JOIN
    (SELECT ... FROM monthly) AS totals), PostgreSQL rejects it because
    subquery aliases aren't reusable like CTEs. This detects the growth
    query pattern specifically and rewrites it to the correct WITH CTE form.
    Triggered when: SQL starts with SELECT, contains LAG(, and references
    a subquery alias in a subsequent FROM clause.
    """
    sql_upper = sql.upper()

    # Only apply to growth/LAG queries that don't already use WITH
    if sql_upper.strip().startswith("WITH"):
        return sql
    if "LAG(" not in sql_upper:
        return sql

    # Detect the broken pattern: FROM (...) AS monthly JOIN (SELECT ... FROM monthly)
    if not re.search(r'FROM\s*\(.*?\)\s*AS\s+monthly', sql, re.IGNORECASE | re.DOTALL):
        return sql

    # Extract ORDER BY direction from the outer query (ASC = decline, DESC = growth)
    direction = "DESC"
    outer_order = re.search(
        r'ORDER\s+BY\s+growth_percent\s+(ASC|DESC)', sql, re.IGNORECASE
    )
    if outer_order:
        direction = outer_order.group(1).upper()

    # Extract the GROUP BY field from inside the subquery to determine
    # whether this is vendor, currency, document_type, etc.
    group_field_match = re.search(
        r"extracted_data->>'(\w+)'\s+AS\s+vendor", sql, re.IGNORECASE
    )
    json_field = group_field_match.group(1) if group_field_match else "vendor_name"

    # Rebuild as proper CTE — correct structure the LLM should have generated
    fixed = f"""WITH monthly AS (
  SELECT
    r.extracted_data->>'{json_field}' AS vendor,
    DATE_TRUNC('month',
      CASE
        WHEN r.extracted_data->>'issue_date' ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'YYYY-MM-DD')
        WHEN r.extracted_data->>'issue_date' ~ '^\\d{{1,2}}/\\d{{1,2}}/\\d{{4}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'DD/MM/YYYY')
        WHEN r.extracted_data->>'issue_date' ~ '^\\d{{1,2}} \\w+ \\d{{4}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'DD Month YYYY')
        WHEN r.extracted_data->>'issue_date' ~ '^\\d{{1,2}}\\.\\d{{1,2}}\\.\\d{{4}}$'
          THEN TO_DATE(r.extracted_data->>'issue_date', 'DD.MM.YYYY')
        ELSE NULL
      END
    ) AS month,
    NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount', '[^0-9.]', '', 'g'), '')::numeric AS amount
  FROM documents d
  JOIN extraction_results r ON r.doc_id = d.id
  WHERE r.document_type = 'invoice'
    AND d.status = 'done'
    AND r.extracted_data->>'vendor_name' IS NOT NULL
),
totals AS (
  SELECT vendor, month, ROUND(SUM(amount), 2) AS total
  FROM monthly
  WHERE month IS NOT NULL
  GROUP BY vendor, month
),
growth AS (
  SELECT
    vendor, month, total,
    LAG(total) OVER (PARTITION BY vendor ORDER BY month) AS prev_total,
    LAG(month) OVER (PARTITION BY vendor ORDER BY month) AS prev_month
  FROM totals
)
SELECT vendor, prev_month, month, prev_total, total,
       ROUND((total - prev_total) / NULLIF(prev_total, 0) * 100, 2) AS growth_percent
FROM growth
WHERE prev_total IS NOT NULL
ORDER BY growth_percent {direction}
LIMIT 1;"""

    logger.info(f"fix_subquery_to_cte: rewrote broken JOIN-subquery to CTE, direction={direction}")
    return fixed

def fix_missing_join_on(sql: str) -> str:
    """
    Fix missing ON clause in JOIN statements.
    JOIN documents d2 JOIN extraction_results r2 ON r2.doc_id = d2.id
    → JOIN documents d2 ON d2.id = r1.doc_id JOIN extraction_results r2 ON r2.doc_id = d2.id
    
    Universal: detects any JOIN without ON and adds the standard doc_id linkage.
    """
    # Pattern: JOIN table alias\nJOIN (missing ON between them)
    fixed = re.sub(
        r'JOIN\s+(documents|extraction_results)\s+(\w+)\s*\n\s*JOIN',
        lambda m: f'JOIN {m.group(1)} {m.group(2)} ON {m.group(2)}.id = r1.doc_id\nJOIN',
        sql,
        flags=re.IGNORECASE
    )
    if fixed != sql:
        logger.info("fix_missing_join_on: added missing ON clause to JOIN")
    return fixed

def fix_jsonb_cast(sql: str) -> str:
    """
    Fix all JSONB ->> cast issues universally:
    1. r.extracted_data->>'field'::text  → (r.extracted_data->>'field')::text
    2. field ILIKE '%' || expr::text || '%'  → ILIKE CONCAT('%', expr, '%')
    3. field ILIKE '%' || expr || '%'  → ILIKE CONCAT('%', expr, '%')
    Avoids asyncpg type inference issues with || and ::text on JSONB results.
    """
    # Fix: any ->> expression followed by ::type cast — wrap in parens
    sql = re.sub(
        r"(\w+\.extracted_data->>'[^']+')\s*::(\w+)",
        r"(\1)::\2",
        sql,
        flags=re.IGNORECASE
    )

    # Fix: ILIKE '%' || any_expr::text || '%' → ILIKE CONCAT('%', any_expr, '%')
    sql = re.sub(
        r"ILIKE\s+'%'\s*\|\|\s*(.+?)\s*\|\|\s*'%'",
        lambda m: f"ILIKE CONCAT('%', {m.group(1).strip()}, '%')",
        sql,
        flags=re.IGNORECASE
    )

    # Fix: ILIKE '%' + (subquery) + '%' → ILIKE CONCAT('%', (subquery), '%')
    sql = re.sub(
        r"ILIKE\s*'%'\s*\+\s*(\(SELECT[^)]+\))\s*\+\s*'%'",
        lambda m: f"ILIKE CONCAT('%', {m.group(1)}, '%')",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Fix: ILIKE '%' + expr + '%' (non-subquery) → ILIKE CONCAT('%', expr, '%')
    sql = re.sub(
        r"ILIKE\s*'%'\s*\+\s*(.+?)\s*\+\s*'%'",
        lambda m: f"ILIKE CONCAT('%', {m.group(1).strip()}, '%')",
        sql,
        flags=re.IGNORECASE
    )

    if "CONCAT" in sql.upper() or ")::" in sql:
        logger.info("fix_jsonb_cast: applied JSONB cast corrections")
    return sql

def fix_cross_document_exists(sql: str, question: str = "") -> str:
    """
    Universal fix for cross-document queries (contracts from vendors who also 
    sent invoices above X, or invoices from vendors who also have contracts, etc).
    
    The LLM typically generates a broken IN(SELECT...) or JOIN pattern that 
    fails because parties is a JSON array. This detects any cross-document
    query involving both invoice and contract document types and rewrites
    to the correct EXISTS + ILIKE CONCAT pattern.

    CRITICAL: "but not in" / "without" / "exclude" intent must be detected
    from the ORIGINAL QUESTION, not the generated SQL — English phrases
    like "but not in invoices" never appear inside SQL code, only in the
    user's natural language question.
    
    Works for any amount, any threshold direction, any document type combination.
    """
    sql_lower = sql.lower()
    question_lower = (question or "").lower()
    # Never rewrite aggregation queries (GROUP BY vendor)
    if "group by" in sql_lower:
        return sql
    # Only applies to cross-document queries
    if "invoice" not in sql_lower or "contract" not in sql_lower:
        return sql
    # Only rewrite if there's an actual amount condition — not generic cross-doc queries
    has_amount_condition = bool(re.search(r'\d+', sql)) and any(
        k in sql_lower for k in [">", "<", ">=", "<=", "above", "over", "more than"]
    )
    if not has_amount_condition:
        return sql

    # Don't rewrite if SQL already has a specific vendor/parties filter
    if "ilike" in sql_lower and "parties" in sql_lower:
        existing_ilike = re.search(r"parties.*?ilike\s*'%([^%]+)%'", sql_lower)
        if existing_ilike:
            return sql  # already has specific vendor filter
    
    # Already correct or uses NOT EXISTS (don't flip the logic)
    if "exists" in sql_lower and "concat" in sql_lower:
        return sql
    if "not exists" in sql_lower:
        return sql
    if "not in" in sql_lower and "exists" in sql_lower:
        return sql
        
    # Already handled by fix_cross_document_vendor_match
    if "exists" in sql_lower and "vendor_name" in sql_lower and "parties" in sql_lower:
        return sql

    # ── Detect NOT EXISTS intent — checked against the QUESTION ──
    not_exists_patterns = [
        r'\bnot\s+in\s+invoices\b',
        r'\bnot\s+in\s+contracts\b',
        r'\bbut\s+not\s+in\b',
        r'\bwithout\s+invoices\b',
        r'\bwithout\s+contracts\b',
        r'\bno\s+invoices\b',
        r'\bno\s+contracts\b',
        r'\bnever\s+invoiced\b',
        r'\bnot\s+appear\s+in\b',
        r'\bdon\'t\s+have\s+invoices\b',
        r'\bdo\s+not\s+have\s+invoices\b',
        r'\bdon\'t\s+have\s+contracts\b',
        r'\bdo\s+not\s+have\s+contracts\b',
        r'\bexclude\b',
        r'\bnot\s+found\s+in\b',
        r'\bonly\s+in\s+contracts\b',
        r'\bonly\s+in\s+invoices\b',
        r'\bmissing\s+from\b',
        r'\bnot\s+also\b',
    ]
    # Search the question if we have it, otherwise fall back to SQL as before
    search_text = question_lower if question_lower else sql_lower
    use_not_exists = any(
        re.search(p, search_text, re.IGNORECASE)
        for p in not_exists_patterns
    )
    exists_keyword = "NOT EXISTS" if use_not_exists else "EXISTS"

    # Extract amount threshold if present
    amount_match = re.search(
        r'(?:>|>=|<|<=)\s*(\d+(?:\.\d+)?)|'
        r'(?:above|over|more than|greater than|exceeding|at least)\s+(\d+(?:\.\d+)?)',
        sql, re.IGNORECASE
    )
    
    direction = ">"
    if amount_match:
        if re.search(r'(?:<|<=|less than|below|under)', sql, re.IGNORECASE):
            direction = "<"
    
    amount = None
    if amount_match:
        amount = amount_match.group(1) or amount_match.group(2)

    currency_match = re.search(
        r"currency\s*=\s*'([A-Z]{3})'|'([A-Z]{3})'",
        sql, re.IGNORECASE
    )
    currency_filter = ""
    if currency_match:
        currency = currency_match.group(1) or currency_match.group(2)
        if currency in ("EUR", "USD", "GBP", "JPY", "CHF", "INR", "CAD", "AUD"):
            currency_filter = f"AND r2.extracted_data->>'currency' = '{currency}'"

    wants_contracts = bool(re.search(
        r"document_type\s*=\s*'contract'", sql[:len(sql)//2], re.IGNORECASE
    ))
    
    if wants_contracts:
        primary_type = "contract"
        filter_type = "invoice"
        primary_fields = """d.filename,
            r.extracted_data->>'parties' as parties,
            r.extracted_data->>'value' as value,
            r.extracted_data->>'currency' as currency"""
        link_condition = "r.extracted_data->>'parties' ILIKE CONCAT('%', r2.extracted_data->>'vendor_name', '%')"
    else:
        primary_type = "invoice"
        filter_type = "contract"
        primary_fields = """d.filename,
            r.extracted_data->>'vendor_name' as vendor,
            r.extracted_data->>'total_amount' as amount,
            r.extracted_data->>'currency' as currency"""
        link_condition = "r2.extracted_data->>'parties' ILIKE CONCAT('%', r.extracted_data->>'vendor_name', '%')"

    if amount:
        amount_filter = f"""AND NULLIF(REGEXP_REPLACE(r2.extracted_data->>'total_amount', '[^0-9.]', '', 'g'), '')::numeric {direction} {amount}"""
    else:
        amount_filter = ""

    fixed = f"""SELECT DISTINCT ON (d.filename)
    {primary_fields}
FROM documents d
JOIN extraction_results r ON r.doc_id = d.id
WHERE r.document_type = '{primary_type}'
AND d.status = 'done'
AND {exists_keyword} (
    SELECT 1
    FROM documents d2
    JOIN extraction_results r2 ON r2.doc_id = d2.id
    WHERE r2.document_type = '{filter_type}'
    AND d2.status = 'done'
    {amount_filter}
    {currency_filter}
    AND {link_condition}
)
ORDER BY d.filename, d.created_at DESC
LIMIT 200"""

    logger.info(f"fix_cross_document_exists: rewrote cross-document query "
                f"primary={primary_type} filter={filter_type} amount={amount} "
                f"dir={direction} exists={exists_keyword}")
    return fixed

# def fix_cross_document_vendor_match(sql: str) -> str:
#     """
#     Detect IN(SELECT parties...) pattern and rewrite to EXISTS+ILIKE.
#     The actual ILIKE cast fix is handled by fix_jsonb_cast.
#     """
#     sql_lower = sql.lower()
#     if "invoice" not in sql_lower or "contract" not in sql_lower:
#         return sql
#     if "parties" not in sql_lower or "vendor_name" not in sql_lower:
#         return sql
#     if "EXISTS" in sql.upper():
#         return sql  # already correct structure, fix_jsonb_cast handles the rest
#     if "IN" not in sql.upper():
#         return sql

#     logger.info("fix_cross_document: rewriting IN(SELECT parties) to EXISTS+ILIKE")
#     return """SELECT DISTINCT r1.extracted_data->>'vendor_name' AS vendor
# FROM extraction_results r1
# JOIN documents d1 ON r1.doc_id = d1.id
# WHERE r1.document_type = 'invoice'
# AND d1.status = 'done'
# AND r1.extracted_data->>'vendor_name' IS NOT NULL
# AND EXISTS (
#   SELECT 1
#   FROM extraction_results r2
#   JOIN documents d2 ON r2.doc_id = d2.id
#   WHERE r2.document_type = 'contract'
#   AND d2.status = 'done'
#   AND r2.extracted_data->>'parties' ILIKE '%' || r1.extracted_data->>'vendor_name' || '%'
# )
# ORDER BY vendor
# LIMIT 200"""

# def fix_cross_document_amount_filter(sql: str) -> str:
#     """
#     Fix queries like "contracts from vendors who also sent invoices above X"
#     These need EXISTS subquery joining invoices with amount filter.
#     Universal — works for any amount threshold.
#     """
#     sql_lower = sql.lower()
#     if "contract" not in sql_lower or "invoice" not in sql_lower:
#         return sql
#     if "exists" in sql_lower:
#         return sql  # already correct

#     # Detect amount threshold from SQL
#     amount_match = re.search(r'(\d+(?:\.\d+)?)', sql)
#     if not amount_match:
#         return sql

#     amount = amount_match.group(1)

#     # Check if this looks like a cross-document filter
#     if "parties" not in sql_lower:
#         return sql

#     logger.info(f"fix_cross_document_amount: rewriting to EXISTS with amount > {amount}")

#     return f"""SELECT DISTINCT ON (d.filename)
#     d.filename,
#     r.extracted_data->>'parties' as parties,
#     r.extracted_data->>'value' as value,
#     r.extracted_data->>'currency' as currency
# FROM documents d
# JOIN extraction_results r ON r.doc_id = d.id
# WHERE r.document_type = 'contract'
# AND d.status = 'done'
# AND EXISTS (
#     SELECT 1
#     FROM documents d2
#     JOIN extraction_results r2 ON r2.doc_id = d2.id
#     WHERE r2.document_type = 'invoice'
#     AND d2.status = 'done'
#     AND NULLIF(REGEXP_REPLACE(r2.extracted_data->>'total_amount', '[^0-9.]', '', 'g'), '')::numeric > {amount}
#     AND r.extracted_data->>'parties' ILIKE CONCAT('%', r2.extracted_data->>'vendor_name', '%')
# )
# ORDER BY d.filename, d.created_at DESC
# LIMIT 200"""

def generate_sql(
    question, history="", 
    resolved_context=None, 
    all_vendors=None, 
    previous_sql=None
    )-> str:
    """Generate SQL from natural language question."""
    context = resolved_context or {}
    amount_ref = context.get("amount_reference", "0") or "0"

    try:
        sql = invoke_with_fallback(
            lambda llm: SQL_PROMPT | llm,
            {
                "question": question,
                "history": history or "No history",
                "resolved_context": str(context),
                "amount_reference": amount_ref,
                "previous_sql": previous_sql or "None" 
            }
        )

        # Clean markdown fences
        if "```" in sql:
            parts = sql.split("```")
            sql = parts[1] if len(parts) > 1 else parts[0]
            if sql.lower().startswith("sql"):
                sql = sql[3:]
        sql = sql.strip()
        sql = fix_missing_join_on(sql)
        sql = fix_jsonb_cast(sql)
        sql = fix_duplicate_null_check(sql)
        sql = fix_distinct_order_conflict(sql)
        sql = fix_vendor_exact_match(sql) 
        if all_vendors:
            sql = fix_vendor_canonical_names(sql, all_vendors)
        sql = fix_spurious_duplicate_check(sql, question)
        sql = fix_missing_currency_filter(sql, question)
        sql = fix_nullif_syntax(sql)
        sql = fix_cross_document_exists(sql, question)
        # sql = fix_cross_document_vendor_match(sql)
        # sql = fix_cross_document_amount_filter(sql)
        sql = fix_subquery_to_cte(sql) 
        sql = fix_outer_subquery_references(sql)

        logger.info(f"Generated SQL: {sql[:200]}")
        return sql

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return ""