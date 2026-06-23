import logging
import os
import re 
print("ORCHESTRATOR FILE:", os.path.abspath(__file__))
import json
import re as _re
from datetime import datetime as _dt
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
# from langchain_groq import ChatGroq
from chatbot.llm_provider import invoke_with_fallback
from langchain_core.prompts import ChatPromptTemplate
from chatbot.response_synthesizer import _build_list_response
from chatbot.special_intent_classifier import classify_special_intent
from chatbot.ranking_selector import parse_ranking_selection, select_from_ranking
from services.vendor_matcher import normalize

from chatbot.memory_manager import (
    save_turn, get_recent_context,
    get_last_metadata, clear_session,
    get_session_stats,
    get_conversation_focus, set_conversation_focus,
    get_last_results, set_last_results,
    get_active_state, set_active_state, update_active_state
)
from chatbot.intent_classifier import classify_intent
from chatbot.context_resolver import resolve_context
from chatbot.query_rewriter import rewrite_query
from chatbot.query_generator import generate_sql, validate_sql
from chatbot.response_synthesizer import (
    synthesize_response, synthesize_error, synthesize_general
)
from services.anomaly_detector import detect_outliers, detect_duplicates
from services.vendor_matcher import find_matches, normalize, find_similar_vendors, similarity
from services.currency_converter import get_exchange_rates, convert_to_base, BASE_CURRENCY

logger = logging.getLogger(__name__)


OPERATION_PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are a query operation planner for a conversational analytics system.

ACTIVE DATASET STATE:
{active_state}

CURRENT QUESTION:
{question}

CONVERSATION HISTORY:
{history}

Your job: decide if this question transforms the active dataset or starts fresh.

Return ONLY valid JSON:
{{
  "type": "transform|standalone",
  "operation": "sort|filter|limit|aggregate|reuse_vendor|remove_limit|null",
  "params": {{
    "field": "amount|created_at|vendor|currency|null",
    "direction": "asc|desc|null",
    "limit": 1,
    "vendor": "extracted from active state if reuse — never invented",
    "currency": "extracted from active state if reuse — never invented",
    "value": "filter value or null"
  }},
  "reuse_state": true or false,
  "reasoning": "one sentence why"
}}

RULES:
- type=transform ONLY when active_state has data AND question modifies it
- type=standalone when question introduces completely new topic
- NEVER invent vendor names — use only from active_state.entities
- operation=sort when question asks for ordering (smallest, largest, latest, oldest, etc.)
- operation=filter when question narrows down current results
- operation=reuse_vendor when question asks about different document type for same vendor
- operation=remove_limit when question asks to show all
- If ambiguous → type=standalone
- operation=reuse_vendor when user asks about a DIFFERENT document type using the same vendor
  Example: user was looking at contracts, now asks "show related invoices" → reuse vendor, switch to invoices
  Example: user was looking at invoices, now asks "contracts from same vendor" → reuse vendor, switch to contracts

Return ONLY JSON. No explanation.
""")

def parse_mixed_date(date_str):
    """
    Parse a date string in any of the mixed formats found across extracted
    documents. Single shared implementation — used by every Step 3 handler
    that needs to map a raw issue_date/start_date string to a real date.
    """
    if not date_str or not str(date_str).strip():
        return None
    s = str(date_str).strip()
    patterns = [
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
        (r'^\d{1,2}/\d{1,2}/\d{4}$', '%d/%m/%Y'),
        (r'^\d{1,2}\.\d{1,2}\.\d{4}$', '%d.%m.%Y'),
        (r'^\d{1,2} [A-Za-z]+ \d{4}$', '%d %B %Y'),
        (r'^\d{1,2} [A-Za-z]{3} \d{4}$', '%d %b %Y'),
    ]
    for regex, fmt in patterns:
        if re.match(regex, s):
            try:
                return _dt.strptime(s, fmt)
            except ValueError:
                continue
    return None


def clean_amount_str(raw):
    """Strip currency symbols/commas, return float or None. Shared utility."""
    if raw is None:
        return None
    cleaned = re.sub(r'[^0-9.]', '', str(raw))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None

# ── Helper: Fetch distinct vendor names from DB ───────────────────────────────

async def get_all_vendor_names(db: AsyncSession) -> list:
    sql = """
        SELECT DISTINCT r.extracted_data->>'vendor_name' as vendor
        FROM extraction_results r
        WHERE r.extracted_data->>'vendor_name' IS NOT NULL
          AND r.extracted_data->>'vendor_name' != ''
    """
    rows = await execute_query(sql, db)
    return [r["vendor"] for r in rows if r.get("vendor")]


# ── Helper: Resolve partial/abbreviated vendor mentions to canonical names ─────

def resolve_vendor_names_in_question(question: str, all_vendors: list) -> str:
    """
    Detect partial/abbreviated vendor mentions and append canonical names
    as a hint for SQL generation — preserves original question structure
    so comparison/context keywords (vs, compare, between) are not lost.
    """
    if not all_vendors:
        return question

    words = question.split()
    found_mappings = {}  # candidate -> canonical

    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            if any(idx in found_mappings for idx in range(i, i + n)):
                continue
            candidate = " ".join(words[i:i + n]).strip(",.?!:;")
            if len(candidate) < 3:
                continue
            # Skip currency codes — never vendor names
            if candidate.upper() in {
                "EUR", "USD", "GBP", "JPY", "CHF",
                "INR", "CAD", "AUD", "CNY", "SEK", "NOK", "DKK"
            }:
                continue
            # Skip currency words (people say "euros", "dollars" — never vendor names)
            currency_words = {
                "euro", "euros", "dollar", "dollars", "pound", "pounds",
                "yen", "rupee", "rupees", "franc", "francs",
                "krona", "kronor", "won", "yuan", "peso", "pesos",
                "rand", "shekel", "shekels", "real", "reais"
            }
            if candidate.lower() in currency_words:
                continue
            # Skip common English words that are never vendor names
            stopwords = {
                "which", "invoice", "invoices", "are", "missing", "due",
                "dates", "date", "the", "all", "show", "find", "get",
                "list", "most", "least", "top", "what", "when", "where",
                "how", "many", "much", "total", "amount", "vendor", "vendors",
                "contract", "contracts", "receipt", "receipts", "between",
                "months", "month", "compare", "versus", "and", "with",
                "from", "for", "this", "that", "have", "has", "any",
                "unusual", "suspicious", "duplicate", "repeated", "grew",
                "declined", "spending", "currency", "type", "types",
                # ── Pronouns — never vendor names, but can fuzzy-match
                # vendor fragments after suffix stripping (e.g. "our" vs "YOUR COMPANY") ──
                "our", "your", "their", "his", "her", "its", "my",
                "we", "you", "they", "it", "us", "them", "i", "he", "she",
                "is", "are", "was", "were", "increasing", "decreasing"
            }
            if candidate.lower() in stopwords:
                continue

            # Single words are riskier for false positives (e.g. "euros" vs
            # "EuroData") — require much higher confidence than multi-word phrases
            fuzzy_threshold = 0.55 if n > 1 else 0.85
            fuzzy = find_matches(candidate, all_vendors, threshold=fuzzy_threshold)
            substring = [v for v in all_vendors if candidate.lower() in v.lower()]
            candidates = list({*fuzzy, *substring})

            if candidates:
                best = max(candidates, key=lambda v: similarity(candidate, v))
                if best.lower() != candidate.lower():
                    found_mappings[i] = (candidate, best)
                    for idx in range(i + 1, i + n):
                        found_mappings[idx] = None  # mark as used

    if not found_mappings:
        return question

    # Build canonical name hint — appended to preserve original structure
    canonical_names = [
        v for v in [
            mapping[1] for mapping in found_mappings.values() if mapping
        ]
    ]

    if not canonical_names:
        return question

    hint = ", ".join(f"'{n}'" for n in canonical_names)
    resolved = f"{question} [vendor names: {hint}]"
    logger.info(f"Vendor hint appended: {resolved}")
    return resolved

# ── Helper: Execute SQL ───────────────────────────────────────────────────────

async def execute_query(sql: str, db: AsyncSession) -> list:
    result = await db.execute(text(sql))
    rows = result.fetchall()
    columns = list(result.keys())
    data = []
    for row in rows:
        row_dict = {}
        for i, col in enumerate(columns):
            val = row[i]
            if hasattr(val, 'isoformat'):
                row_dict[col] = val.isoformat()
            elif val is None:
                row_dict[col] = None
            elif isinstance(val, Decimal):
                row_dict[col] = float(val)
            elif isinstance(val, (int, float, bool)):
                row_dict[col] = val
            else:
                row_dict[col] = str(val)
        data.append(row_dict)
    return data

def deduplicate_results(results: list) -> list:
    """
    Remove duplicate rows using filename as primary key.
    Only deduplicate if filename exists — never collapse real unique rows.
    """
    seen = set()
    deduped = []
    for row in results:
        filename = row.get("filename", "")
        if filename:
            # Use filename as unique key — most reliable identifier
            if filename not in seen:
                seen.add(filename)
                deduped.append(row)
        else:
            # No filename — aggregation row, always keep
            deduped.append(row)
    return deduped

# ── Helper: Complexity Check ──────────────────────────────────────────────────

def is_complex_question(question: str) -> bool:
    q = question.lower()

    # ── Trend/grouping questions are ALWAYS a single GROUP BY query ──
    # Decomposing these loses the vendor filter and date-parsing logic
    single_query_patterns = [
        "monthly trend", "trend over", "invoice trend",
        "trend for", "by month", "grouped by month",
        "month over month", "monthly breakdown",
        "trend by currency", "trend by vendor", "trend by",
    ]
    if any(p in q for p in single_query_patterns):
        return False

    complexity_indicators = [
        "and then", "also show", "summarize",
        "as well as", "additionally", "furthermore",
        "compare and", "show me both"
    ]
    has_indicator = any(ind in q for ind in complexity_indicators)
    has_many_commas = question.count(",") > 2
    return has_indicator or has_many_commas


# ── Helper: Question Decomposer ───────────────────────────────────────────────

def decompose_question(question: str) -> list:
    prompt = ChatPromptTemplate.from_template("""
Break this complex question into 2-4 simple atomic sub-questions.
Each sub-question must be answerable with a single SQL query.
Return ONLY a numbered list. No explanation.

Question: {question}

Sub-questions:
""")
    try:
        content = invoke_with_fallback(
            lambda llm: prompt | llm,
            {"question": question}
        )
        lines = content.split("\n")
        sub_questions = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                cleaned = line.split(".", 1)[-1].split(")", 1)[-1].strip()
                if cleaned:
                    sub_questions.append(cleaned)
        return sub_questions[:4] if sub_questions else [question]
    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        return [question]


# ── Helper: Build Conversation Focus ─────────────────────────────────────────

def build_conversation_focus(
    intent: str,
    question: str,
    results: list,
    resolved_context: dict,
    previous_focus: dict
) -> dict:
    q = question.lower()
    focus = {
        "topic": previous_focus.get("topic", intent),
        "query_type": previous_focus.get("query_type", intent),
        "result_count": len(results)
    }
    if results:
        first = results[0]
        if first.get("vendor"):
            focus["vendor"] = first["vendor"]
        if first.get("currency"):
            focus["currency"] = first["currency"]
        if first.get("amount"):
            focus["amount"] = str(first["amount"])
        if first.get("filename"):
            focus["filename"] = first["filename"]
        if first.get("document_type"):
            focus["doc_type"] = first["document_type"]

    if any(k in q for k in ["duplicate", "repeat", "same invoice", "appears", "how many times"]):
        focus["topic"] = "duplicate_invoices"
        focus["query_type"] = "duplicate_detection"
    elif any(k in q for k in ["overdue", "past due", "late payment", "unpaid"]):
        focus["topic"] = "overdue_invoices"
        focus["query_type"] = "date_filter"
    elif any(k in q for k in ["expir", "ending soon", "renewal", "end date"]):
        focus["topic"] = "expiring_contracts"
        focus["query_type"] = "date_filter"
    elif any(k in q for k in ["top vendor", "highest vendor", "most paid", "vendor total"]):
        focus["topic"] = "vendor_analytics"
        focus["query_type"] = "aggregation"
    elif any(k in q for k in ["anomal", "suspicious", "unusual", "outlier", "fraud"]):
        focus["topic"] = "anomaly_detection"
        focus["query_type"] = "anomaly"
    elif any(k in q for k in ["compare", " vs ", "versus", "this month", "last month"]):
        focus["topic"] = "comparison"
        focus["query_type"] = "comparison"
    elif any(k in q for k in ["total", "sum", "average", "count", "how much"]):
        focus["topic"] = "aggregation"
        focus["query_type"] = "aggregation"
    elif any(k in q for k in ["contract", "agreement", "vertrag"]):
        focus["topic"] = "contract_list"
        focus["query_type"] = "list"
    elif any(k in q for k in ["invoice", "bill", "rechnung"]):
        focus["topic"] = "invoice_list"
        focus["query_type"] = "list"

    return focus


# ── NEW: Query Operation Planner ──────────────────────────────────────────────

def plan_query_operation(
    question: str,
    active_state: dict,
    resolved_context: dict,
    history: str = ""
) -> dict:
    """
    Universal operation planner using LLM semantic reasoning.
    No hardcoded keyword lists — works for any language, any phrasing.
    """
    # Skip LLM if no active state — always standalone
    if not active_state.get("active_dataset") and not active_state.get("last_sql"):
        return {"type": "standalone", "operation": None, "params": {}, "reuse_state": False}

    try:
        content = invoke_with_fallback(
            lambda llm: OPERATION_PLANNER_PROMPT | llm,
            {
                "question": question,
                "active_state": json.dumps(active_state, default=str),
                "history": history or "No history"
            }
        )
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        operation = json.loads(content.strip())

        # ── Normalize literal "null"/"None" strings to real None ──
        # The LLM sometimes returns "null" as a JSON string instead of
        # actual JSON null, which breaks every downstream `is None` /
        # `not in (..., None)` check. Fix it once, here, for every field.
        def _normalize_nulls(obj):
            if isinstance(obj, dict):
                return {k: _normalize_nulls(v) for k, v in obj.items()}
            if isinstance(obj, str) and obj.strip().lower() in ("null", "none"):
                return None
            return obj

        operation = _normalize_nulls(operation)

        logger.info(f"Operation plan: {operation}")
        return operation

    except Exception as e:
        logger.error(f"Operation planning failed: {e}")
        return {"type": "standalone", "operation": None, "params": {}, "reuse_state": False}

    # ── Detect transformation operations semantically ──

    # Sort operations
    if any(k in q for k in ["smallest", "cheapest", "lowest", "minimum"]):
        return {
            "type": "transform",
            "operation": "sort",
            "params": {"field": "amount", "direction": "asc", "limit": 1},
            "reuse_state": True
        }

    if any(k in q for k in ["largest", "biggest", "highest", "maximum", "most expensive"]):
        return {
            "type": "transform",
            "operation": "sort",
            "params": {"field": "amount", "direction": "desc", "limit": 1},
            "reuse_state": True
        }

    if any(k in q for k in ["latest", "newest", "most recent"]):
        return {
            "type": "transform",
            "operation": "sort",
            "params": {"field": "created_at", "direction": "desc", "limit": 1},
            "reuse_state": True
        }

    if any(k in q for k in ["oldest", "earliest", "first"]):
        return {
            "type": "transform",
            "operation": "sort",
            "params": {"field": "created_at", "direction": "asc", "limit": 1},
            "reuse_state": True
        }

    # Show all — remove pagination/limit
    if q in ["show all", "show me all", "all of them", "give me all", "list all"]:
        return {
            "type": "transform",
            "operation": "remove_limit",
            "params": {},
            "reuse_state": True
        }

    # Vendor filter — "only BrightPath", "just that vendor"
    vendor_from_context = resolved_context.get("vendor_name") or active_vendor
    if vendor_from_context and any(k in q for k in ["only", "just", "from that vendor", "same vendor"]):
        return {
            "type": "transform",
            "operation": "filter",
            "params": {"vendor": vendor_from_context},
            "reuse_state": True
        }

    # Currency filter — "only EUR", "in USD"
    currency_from_context = resolved_context.get("currency")
    if currency_from_context and any(k in q for k in ["only", "just", "in ", "currency"]):
        return {
            "type": "transform",
            "operation": "filter",
            "params": {"currency": currency_from_context},
            "reuse_state": True
        }

    # Cross-document reuse — "contracts from same vendor"
    if active_vendor and any(k in q for k in ["contract", "receipt", "report"]) and \
       any(k in q for k in ["same vendor", "this vendor", "that vendor"]):
        new_dataset = "contracts" if "contract" in q else \
                      "receipts" if "receipt" in q else "reports"
        return {
            "type": "transform",
            "operation": "reuse_vendor",
            "params": {"vendor": active_vendor, "dataset": new_dataset},
            "reuse_state": True
        }

    # Default — standalone new query
    return {"type": "standalone", "operation": None, "params": {}, "reuse_state": False}


def apply_operation_to_sql(
    base_sql: str,
    operation: dict,
    active_state: dict
) -> str:
    """
    Apply a transformation operation to existing SQL.
    Returns modified SQL without calling LLM again.
    """
    op = operation.get("operation")
    params = operation.get("params", {})
    # Strip trailing semicolon — prevents "; LIMIT 200" syntax error
    base_sql = base_sql.rstrip().rstrip(';').strip()

    if op == "sort":
        field = params.get("field", "amount")
        direction = params.get("direction", "desc")
        limit = params.get("limit", 1)
        # Remove existing ORDER BY and LIMIT
        sql = base_sql.upper()
        clean = base_sql
        if "ORDER BY" in sql:
            clean = base_sql[:base_sql.upper().rfind("ORDER BY")]
        if "LIMIT" in sql:
            clean = clean[:clean.upper().rfind("LIMIT")]
        # Map field to actual SQL expression
        field_map = {
            "amount": "NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric",
            "created_at": "d.created_at"
        }
        sql_field = field_map.get(field, field)
        return f"{clean.strip()} ORDER BY {sql_field} {direction.upper()} NULLS LAST LIMIT {limit}"

    if op == "filter":
        clean = base_sql
        order_clause = ""
        sql_upper = clean.upper()

        if "ORDER BY" in sql_upper:
            order_pos = clean.upper().rfind("ORDER BY")
            order_clause = clean[order_pos:]
            clean = clean[:order_pos].strip()
            if "LIMIT" in order_clause.upper():
                limit_pos = order_clause.upper().rfind("LIMIT")
                order_clause = order_clause[:limit_pos].strip()
        elif "LIMIT" in sql_upper:
            clean = clean[:clean.upper().rfind("LIMIT")].strip()

        vendor = params.get("vendor")
        currency = params.get("currency")
        amount_val = params.get("value")

        if vendor and str(vendor).lower() not in ("null", "none", ""):
            vendor = str(vendor).replace("'", "''")
            clean += f" AND LOWER(r.extracted_data->>'vendor_name') = LOWER('{vendor}')"

        # ── Only add currency if explicitly requested, not from active state ──
        if currency and str(currency).lower() not in ("null", "none", "") and "only" in params.get("original_question", "").lower():
            currency = str(currency).replace("'", "''")
            clean += f" AND r.extracted_data->>'currency' = '{currency}'"

        if amount_val and str(amount_val).lower() not in ("null", "none", ""):
            try:
                amount_num = float(str(amount_val).replace(",", ""))
                clean += f" AND NULLIF(REPLACE(r.extracted_data->>'total_amount',',',''),'')::numeric > {amount_num}"
            except ValueError:
                pass

        if order_clause:
            clean += f" {order_clause}"
        clean += " LIMIT 200"
        return clean

    if op == "remove_limit":
        sql = base_sql.upper()
        if "LIMIT" in sql:
            return base_sql[:base_sql.upper().rfind("LIMIT")] + "LIMIT 200"
        return base_sql

    if op == "reuse_vendor":
        vendor = params.get("vendor", "")
        # Determine target document type from question
        dataset = params.get("dataset", "contract")
        doc_type = "contract" if "contract" in dataset else \
                   "receipt" if "receipt" in dataset else \
                   "invoice"

        if not vendor or str(vendor).lower() in ("null", "none", ""):
            return base_sql

        vendor = str(vendor).replace("'", "''")

        if doc_type == "contract":
            return f"""
                    SELECT DISTINCT ON (d.filename) d.filename,
                        r.extracted_data->>'parties' as parties,
                        r.extracted_data->>'value' as value,
                        r.extracted_data->>'currency' as currency,
                        r.extracted_data->>'start_date' as start_date,
                        r.extracted_data->>'end_date' as end_date
                    FROM documents d
                    JOIN extraction_results r ON r.doc_id = d.id
                    WHERE r.document_type = 'contract'
                    AND d.status = 'done'
                    AND r.extracted_data->>'parties' ILIKE '%{vendor}%'
                    ORDER BY d.filename, d.created_at DESC
                    LIMIT 200
                    """
        else:
            return f"""
                    SELECT DISTINCT ON (d.filename) d.filename,
                        r.extracted_data->>'vendor_name' as vendor,
                        NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric as amount,
                        r.extracted_data->>'currency' as currency
                    FROM documents d
                    JOIN extraction_results r ON r.doc_id = d.id
                    WHERE r.document_type = '{doc_type}'
                    AND d.status = 'done'
                    AND LOWER(r.extracted_data->>'vendor_name') = LOWER('{vendor}')
                    ORDER BY d.filename, d.created_at DESC
                    LIMIT 200
                    """

    return base_sql


# ── Main Orchestrator ─────────────────────────────────────────────────────────

async def process_message(
    question: str,
    session_id: str,
    db: AsyncSession
) -> dict:
    logger.info(f"[{session_id}] Processing: {question}")

    # ── Step 1: Load memory ───────────────────────
    history = get_recent_context(session_id)
    last_metadata = get_last_metadata(session_id)
    conversation_focus = get_conversation_focus(session_id)
    last_results = get_last_results(session_id)
    active_state = get_active_state(session_id)

    logger.info(f"[{session_id}] Focus: {conversation_focus}")

    save_turn(session_id, "user", question)

    # ── Step 2: Classify intent ───────────────────
    intent = classify_intent(question, history)
    logger.info(f"[{session_id}] Intent: {intent}")

    # ── Step 2.5: Detect degenerate self-comparison (X vs X) ──────────────────
    self_compare_match = re.search(
        r'\b([A-Za-z][A-Za-z0-9\s]{1,40}?)\s+vs\.?\s+\1\b',
        question, re.IGNORECASE
    )
    if not self_compare_match:
        # Also catch "compare X and X" / "compare X with X" phrasing
        self_compare_match = re.search(
            r'compare\s+([A-Za-z][A-Za-z0-9\s]{1,40}?)\s+(?:and|with)\s+\1\b',
            question, re.IGNORECASE
        )
    if self_compare_match:
        item = self_compare_match.group(1).strip()
        answer = f"\"{item}\" and \"{item}\" are the same — there's nothing to compare."
        save_turn(session_id, "assistant", answer, {"intent": "self_comparison"})
        return {
            "answer": answer,
            "intent": "self_comparison",
            "session_id": session_id,
            "rewritten": None,
            "sql": None,
            "results": [],
            "count": 0
        }

    # ── Step 3a: Handle reset ─────────────────────
    if intent == "reset":
        clear_session(session_id)
        return {
            "answer": "Conversation cleared! Starting fresh.",
            "intent": "reset",
            "session_id": session_id,
            "rewritten": None,
            "sql": None,
            "results": [],
            "count": 0
        }

    # ── Step 3b: Handle greeting / general ───────
    if intent in ["greeting", "general"]:
        answer = synthesize_general(question, history)
        save_turn(session_id, "assistant", answer, {
            "intent": intent,
            "original_question": question
        })
        return {
            "answer": answer,
            "intent": intent,
            "session_id": session_id,
            "rewritten": None,
            "sql": None,
            "results": [],
            "count": 0
        }

    # ── Step 3c: Handle clarification ────────────
    if intent == "clarification":
        answer = synthesize_response(question, question, [], history, "clarification")
        save_turn(session_id, "assistant", answer, {"intent": intent})
        return {
            "answer": answer,
            "intent": intent,
            "session_id": session_id,
            "rewritten": None,
            "sql": None,
            "results": [],
            "count": 0
        }

    # ── Step 3d: Handle vendor fuzzy matching ─────
    fuzzy_keywords = [
        "slightly different", "similar spelling", "same vendor",
        "duplicate vendor", "vendor variations", "similar vendors",
        "vendor spelling", "same company", "vendor duplicates"
    ]
    if any(kw in question.lower() for kw in fuzzy_keywords):
        logger.info(f"[{session_id}] Vendor fuzzy matching triggered")
        fuzzy_sql = """
            SELECT DISTINCT r.extracted_data->>'vendor_name' as vendor
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE d.status = 'done'
            AND r.extracted_data->>'vendor_name' IS NOT NULL
            LIMIT 200
        """
        is_valid, _ = validate_sql(fuzzy_sql)
        if is_valid:
            try:
                all_vendors_data = await execute_query(fuzzy_sql, db)
                all_vendors = [v["vendor"] for v in all_vendors_data if v["vendor"]]
                groups = find_similar_vendors(all_vendors)
                if groups:
                    lines = ["These vendors may be the same entity:\n"]
                    for group in groups:
                        lines.append(f"• {' / '.join(group)}")
                    answer = "\n".join(lines)
                else:
                    answer = "No similar vendor spellings found. All vendor names appear unique."
                save_turn(session_id, "assistant", answer, {
                    "intent": "vendor_fuzzy",
                    "sql": fuzzy_sql,
                    "count": len(all_vendors_data)
                })
                return {
                    "answer": answer,
                    "intent": "vendor_fuzzy",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": fuzzy_sql,
                    "results": all_vendors_data,
                    "count": len(all_vendors_data)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Vendor fuzzy matching failed: {e}", exc_info=True)

    # ── Step 3e: Handle repeat/duplicate detection ─
    repeat_keywords = [
        "repeat", "repeated", "appears again", "same invoice",
        "invoice repeat", "which invoice repeat", "most repeated",
        "duplicate invoice", "duplicate invoices", "find duplicate"
    ]
    # Skip if question has extra filters — let Step 4 decompose it properly
    _q = question.lower()
    has_extra_filters = any([
        any(c in _q for c in ["eur", "usd", "gbp", "jpy", "chf", "inr"]),
        bool(re.search(r'\b(above|below|over|under|more than|less than|greater than)\s+\d+', _q)),
        any(k in _q for k in ["oldest", "newest", "latest", "earliest", "highest", "lowest"]),
        any(k in _q for k in ["show all", "only", "from vendor", "and show", "and find"]),
        any(k in _q for k in ["expiring", "expiring contracts", "who also", "that also"]),  # ← ADD THIS
    ])

    if any(kw in question.lower() for kw in repeat_keywords) and not has_extra_filters:
        logger.info(f"[{session_id}] Repeat invoice detection triggered")

        q_lower = question.lower()
        is_most = any(k in q_lower for k in [
            "most", "highest", "top", "which one", "maximum", "max"
        ])
        is_least = any(k in q_lower for k in [
            "least", "less", "lowest", "minimum", "min", "fewest", "rarest"
        ])
        repeat_order = "ASC" if is_least else "DESC"
        repeat_limit = "LIMIT 20"

        repeat_sql = f"""
            SELECT
                r.extracted_data->>'invoice_number' as invoice_number,
                MAX(r.extracted_data->>'vendor_name') as vendor,
                MAX(r.extracted_data->>'total_amount') as amount,
                MAX(r.extracted_data->>'currency') as currency,
                COUNT(*) as repeat_count
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE d.status = 'done'
            AND r.document_type = 'invoice'
            AND r.extracted_data->>'invoice_number' IS NOT NULL
            AND r.extracted_data->>'invoice_number' != ''
            GROUP BY
                r.extracted_data->>'invoice_number'
            HAVING COUNT(*) > 1
            ORDER BY repeat_count {repeat_order}
            {repeat_limit}
        """
        is_valid, _ = validate_sql(repeat_sql)
        if is_valid:
            try:
                repeat_results = await execute_query(repeat_sql, db)
                if repeat_results:
                    # For "most/highest" queries, show all invoices tied at the top count
                    if is_most or is_least:
                        top_count = repeat_results[0].get("repeat_count")
                        repeat_results = [
                            r for r in repeat_results
                            if str(r.get("repeat_count")) == str(top_count)
                        ]

                    lines = []
                    for i, r in enumerate(repeat_results[:10], 1):
                        lines.append(
                            f"{i}. {r.get('invoice_number', 'unknown')} | "
                            f"{r.get('vendor', 'unknown')} | "
                            f"{r.get('repeat_count')} times"
                        )
                    answer = "\n".join(lines)
                    if len(repeat_results) > 10:
                        answer += f"\n... and {len(repeat_results) - 10} more"
                else:
                    answer = "No repeated invoices found."

                new_focus = build_conversation_focus(
                    "duplicate_detection", question, repeat_results, {}, conversation_focus
                )
                new_focus["topic"] = "duplicate_invoices"
                new_focus["query_type"] = "duplicate_detection"
                set_conversation_focus(session_id, new_focus)
                set_last_results(session_id, repeat_results)

                update_active_state(session_id, {
                    "intent": "duplicate_detection",
                    "active_dataset": "invoices",
                    "filters": {},
                    "result_count": len(repeat_results),
                    "last_sql": repeat_sql
                })

                save_turn(session_id, "assistant", answer, {
                    "intent": "duplicate_detection",
                    "sql": repeat_sql,
                    "count": len(repeat_results),
                    "results_sample": repeat_results[:3]
                })
                return {
                    "answer": answer,
                    "intent": "duplicate_detection",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": repeat_sql,
                    "results": repeat_results,
                    "count": len(repeat_results)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Repeat detection failed: {e}", exc_info=True)

    # ── Step 3f: Handle overdue/expiring with safe Python SQL ────
    overdue_keywords = ["overdue", "past due", "late payment", "unpaid", "outstanding"]
    expiring_keywords = ["expiring", "expiring soon", "ending soon", "renewal due"]

    _q_overdue = question.lower()
    has_overdue_extra = any([
        any(k in _q_overdue for k in [
            "expiring", "expiring contracts", "with expiring",
            "who also have", "that also have", "who have contracts",
            "and contracts", "with contracts"
        ]),
        ("contract" in _q_overdue and any(k in _q_overdue for k in [
            "who also", "that also", "also have"
        ])),
    ])

    if any(kw in _q_overdue for kw in overdue_keywords) and not has_overdue_extra:
        logger.info(f"[{session_id}] Overdue invoice detection triggered")
        overdue_sql = r"""
            SELECT DISTINCT ON (d.filename)
                d.filename,
                r.extracted_data->>'vendor_name' as vendor,
                r.extracted_data->>'due_date' as due_date,
                r.extracted_data->>'total_amount' as amount,
                r.extracted_data->>'currency' as currency
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE r.document_type = 'invoice'
            AND d.status = 'done'
            AND r.extracted_data->>'due_date' IS NOT NULL
            AND r.extracted_data->>'due_date' != ''
            AND (
                CASE
                    WHEN r.extracted_data->>'due_date' ~ '^\d{4}-\d{2}-\d{2}$'
                    THEN TO_DATE(r.extracted_data->>'due_date', 'YYYY-MM-DD')
                    WHEN r.extracted_data->>'due_date' ~ '^\d{1,2}/\d{1,2}/\d{4}$'
                    THEN TO_DATE(r.extracted_data->>'due_date', 'DD/MM/YYYY')
                    WHEN r.extracted_data->>'due_date' ~ '^\d{1,2}\.\d{1,2}\.\d{4}$'
                    THEN TO_DATE(r.extracted_data->>'due_date', 'DD.MM.YYYY')
                    WHEN r.extracted_data->>'due_date' ~ '^\d{1,2} \w+ \d{4}$'
                    THEN TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY')
                    ELSE NULL
                END
            ) < CURRENT_DATE
            ORDER BY d.filename
            LIMIT 200
        """
        is_valid, _ = validate_sql(overdue_sql)
        if is_valid:
            try:
                overdue_results = await execute_query(overdue_sql, db)
                answer = _build_list_response(overdue_results, question) if overdue_results else "No overdue invoices found."
                new_focus = build_conversation_focus("overdue", question, overdue_results, {}, conversation_focus)
                set_conversation_focus(session_id, new_focus)
                set_last_results(session_id, overdue_results)
                update_active_state(session_id, {
                    "intent": "overdue_invoices",
                    "active_dataset": "invoice",
                    "result_count": len(overdue_results),
                    "last_sql": overdue_sql
                })
                save_turn(session_id, "assistant", answer, {
                    "intent": "overdue_invoices",
                    "count": len(overdue_results),
                    "results_sample": overdue_results[:3]
                })
                return {
                    "answer": answer,
                    "intent": "overdue_invoices",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": overdue_sql,
                    "results": overdue_results,
                    "count": len(overdue_results)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Overdue detection failed: {e}", exc_info=True)

    _q_expiring = question.lower()
    has_expiring_extra = any([
        any(k in _q_expiring for k in [
            "overdue", "past due", "unpaid", "late payment",
            "with overdue", "and overdue", "from vendors who"
        ]),
        ("invoice" in _q_expiring and any(k in _q_expiring for k in [
            "who also", "that also", "also have"
        ])),
    ])

    if any(kw in _q_expiring for kw in expiring_keywords) and not has_expiring_extra:
        logger.info(f"[{session_id}] Expiring contract detection triggered")
        expiring_sql = r"""
            SELECT DISTINCT ON (d.filename)
                d.filename,
                r.extracted_data->>'parties' as parties,
                r.extracted_data->>'end_date' as end_date,
                r.extracted_data->>'value' as value,
                r.extracted_data->>'currency' as currency
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE r.document_type = 'contract'
            AND d.status = 'done'
            AND r.extracted_data->>'end_date' IS NOT NULL
            AND r.extracted_data->>'end_date' != ''
            AND (
                CASE
                    WHEN r.extracted_data->>'end_date' ~ '^\d{4}-\d{2}-\d{2}$'
                    THEN TO_DATE(r.extracted_data->>'end_date', 'YYYY-MM-DD')
                    WHEN r.extracted_data->>'end_date' ~ '^\d{1,2}/\d{1,2}/\d{4}$'
                    THEN TO_DATE(r.extracted_data->>'end_date', 'DD/MM/YYYY')
                    WHEN r.extracted_data->>'end_date' ~ '^\d{1,2}\.\d{1,2}\.\d{4}$'
                    THEN TO_DATE(r.extracted_data->>'end_date', 'DD.MM.YYYY')
                    WHEN r.extracted_data->>'end_date' ~ '^\d{1,2} \w+ \d{4}$'
                    THEN TO_DATE(r.extracted_data->>'end_date', 'DD Month YYYY')
                    ELSE NULL
                END
            ) BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
            ORDER BY d.filename
            LIMIT 200
        """
        is_valid, _ = validate_sql(expiring_sql)
        if is_valid:
            try:
                expiring_results = await execute_query(expiring_sql, db)
                answer = _build_list_response(expiring_results, question) if expiring_results else "No contracts expiring in the next 30 days."
                new_focus = build_conversation_focus("expiring", question, expiring_results, {}, conversation_focus)
                set_conversation_focus(session_id, new_focus)
                set_last_results(session_id, expiring_results)
                save_turn(session_id, "assistant", answer, {
                    "intent": "expiring_contracts",
                    "count": len(expiring_results),
                    "results_sample": expiring_results[:3]
                })
                return {
                    "answer": answer,
                    "intent": "expiring_contracts",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": expiring_sql,
                    "results": expiring_results,
                    "count": len(expiring_results)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Expiring detection failed: {e}", exc_info=True)

    # ── Step 3g: Handle total invoice amount by currency ─────
    total_keywords = [
        "total amount of all invoices", "total invoices amount",
        "grand total of all", "overall total of all",
        "sum of all invoices", "total amount all invoices",
        "how much did we spend", "how much money", "what did we spend",
        "how much have we paid", "how much total"
    ]
    if any(kw in question.lower() for kw in total_keywords):
        logger.info(f"[{session_id}] Total invoice amount by currency triggered")
        total_sql = """
            SELECT
                r.extracted_data->>'currency' as currency,
                ROUND(SUM(
                    NULLIF(
                        REGEXP_REPLACE(
                            r.extracted_data->>'total_amount',
                            '[^0-9.]', '', 'g'
                        ), ''
                    )::numeric
                ), 2) as total_amount,
                COUNT(*) as invoice_count
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE r.document_type = 'invoice'
            AND d.status = 'done'
            AND r.extracted_data->>'currency' IS NOT NULL
            GROUP BY currency
            ORDER BY total_amount DESC
        """
        

        is_valid, _ = validate_sql(total_sql)
        if is_valid:
            try:
                total_results = await execute_query(total_sql, db)
                if total_results:
                    lines = ["Invoices span multiple currencies. Totals per currency:\n"]
                    for row in total_results:
                        try:
                            amount = f"{float(row.get('total_amount', 0)):,.2f}"
                        except (ValueError, TypeError):
                            amount = str(row.get('total_amount', 0))
                        lines.append(
                            f"• {row.get('currency')} — "
                            f"{amount} "
                            f"({row.get('invoice_count')} invoices)"
                        )
                    answer = "\n".join(lines)
                else:
                    answer = "No invoice totals found."

                save_turn(session_id, "assistant", answer, {
                    "intent": "total_by_currency",
                    "count": len(total_results),
                    "results_sample": total_results[:3]
                })
                return {
                    "answer": answer,
                    "intent": "total_by_currency",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": total_sql,
                    "results": total_results,
                    "count": len(total_results)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Total by currency failed: {e}", exc_info=True)

    # ── Step 3h: Handle currency comparison queries ───────
    currency_keywords = [
        "which currency has the most", "currency breakdown",
        "spending by currency", "total by currency",
        "which currency is highest", "highest currency total"
    ]
    if any(kw in question.lower() for kw in currency_keywords):
        logger.info(f"[{session_id}] Currency breakdown triggered")
        currency_sql = """
            SELECT
                r.extracted_data->>'currency' as currency,
                ROUND(SUM(
                    NULLIF(
                        REGEXP_REPLACE(
                            r.extracted_data->>'total_amount',
                            '[^0-9.]', '', 'g'
                        ), ''
                    )::numeric
                ), 2) as total_amount,
                COUNT(*) as invoice_count
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE r.document_type = 'invoice'
            AND d.status = 'done'
            AND r.extracted_data->>'currency' IS NOT NULL
            GROUP BY currency
            ORDER BY total_amount DESC
        """
        is_valid, _ = validate_sql(currency_sql)
        if is_valid:
            try:
                currency_results = await execute_query(currency_sql, db)
                if currency_results:
                    top = currency_results[0]
                    try:
                        top_amount = f"{float(top.get('total_amount', 0)):,.2f}"
                    except (ValueError, TypeError):
                        top_amount = str(top.get('total_amount', 0))

                    lines = [
                        f"USD leads with the highest total. Full breakdown:\n"
                        if top.get('currency') == 'USD'
                        else f"{top.get('currency')} has the highest total at {top_amount}. Full breakdown:\n"
                    ]
                    for row in currency_results:
                        try:
                            amount = f"{float(row.get('total_amount', 0)):,.2f}"
                        except (ValueError, TypeError):
                            amount = str(row.get('total_amount', 0))
                        lines.append(
                            f"• {row.get('currency')} — {amount} "
                            f"({row.get('invoice_count')} invoices)"
                        )
                    answer = "\n".join(lines)
                else:
                    answer = "No currency data found."

                save_turn(session_id, "assistant", answer, {
                    "intent": "currency_breakdown",
                    "count": len(currency_results),
                    "results_sample": currency_results[:3]
                })
                return {
                    "answer": answer,
                    "intent": "currency_breakdown",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": currency_sql,
                    "results": currency_results,
                    "count": len(currency_results)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Currency breakdown failed: {e}", exc_info=True)
                try:
                    await db.rollback()
                except Exception:
                    pass

    # ── Step 3v: Semantic special-intent routing — understands MEANING,
    # not exact phrasing. Computed once, used by Steps 3i, 3x, 3y, 3z, 3w below. ──
    special_intent = classify_special_intent(question, history)
    logger.info(f"[{session_id}] Special intent: {special_intent}")

    # ── Step 3i: Cross-currency "biggest bills" — real exchange rate conversion ──
    biggest_keywords = [
        "biggest bill", "biggest invoice", "biggest bills", "biggest invoices",
        "largest bill", "largest invoices", "largest bills",
        "what are my biggest", "what are my largest",
        "most expensive bill", "most expensive bills"
    ]
    if special_intent == "biggest_bills":
        logger.info(f"[{session_id}] Cross-currency biggest bills triggered")
        raw_sql = """
            SELECT DISTINCT ON (d.filename)
                d.filename,
                r.extracted_data->>'vendor_name' as vendor,
                NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric as amount,
                r.extracted_data->>'currency' as currency,
                r.extracted_data->>'issue_date' as issue_date
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE r.document_type = 'invoice'
            AND d.status = 'done'
            AND r.extracted_data->>'total_amount' IS NOT NULL
            ORDER BY d.filename, d.created_at DESC
        """
        is_valid, _ = validate_sql(raw_sql)
        if is_valid:
            try:
                raw_results = await execute_query(raw_sql, db)
                rates = await get_exchange_rates()

                converted = []
                for row in raw_results:
                    amt = row.get("amount")
                    cur = row.get("currency")
                    if amt is None or not cur:
                        continue
                    base_amt = convert_to_base(float(amt), cur, rates)
                    if base_amt is None:
                        continue
                    converted.append({**row, "base_amount": round(base_amt, 2)})

                converted.sort(key=lambda r: r["base_amount"], reverse=True)

                ranked_bills = [(i, row["base_amount"]) for i, row in enumerate(converted)]
                bills_by_index = {i: row for i, row in enumerate(converted)}

                selection = parse_ranking_selection(question)
                show_month = selection.get("show_month", False)

                def format_bill_line(idx_key, score, idx=None):
                    row = bills_by_index[idx_key]
                    prefix = f"{idx}. " if idx else "• "
                    try:
                        amount_val = float(row['amount'])
                    except (ValueError, TypeError):
                        amount_val = 0.0
                    line = (
                        f"{prefix}{row['filename']} | {row.get('vendor', 'Unknown')} | "
                        f"{amount_val:,.2f} {row['currency']} "
                        f"(≈ {row['base_amount']:,.2f} {BASE_CURRENCY})"
                    )
                    if show_month:
                        line += f" | issued: {row.get('issue_date') or 'unknown date'}"
                    return line

                result = select_from_ranking(ranked_bills, selection, format_bill_line)
                top_results = [bills_by_index[idx_key] for idx_key, _ in result["selected"]]

                if show_month and top_results:
                    # Lead with the month — that's what the user actually asked for.
                    dt = parse_mixed_date(top_results[0].get("issue_date"))
                    ordinal = {1: "highest", -1: "lowest"}.get(selection.get("rank_position"), "selected")
                    pos = selection.get("rank_position")
                    if pos and pos > 1:
                        ordinal_map = {2: "second highest", 3: "third highest"}
                        ordinal = ordinal_map.get(pos, f"{pos}th highest")
                    elif pos and pos < -1:
                        ordinal_map = {-2: "second lowest", -3: "third lowest"}
                        ordinal = ordinal_map.get(pos, f"{abs(pos)}th lowest")

                    if dt:
                        month_label = dt.strftime("%B %Y")
                        lines = [f"📅 The {ordinal} invoice was issued in **{month_label}**.\n"]
                    else:
                        lines = [f"📅 Could not determine the month — issue date is missing or unparseable.\n"]

                    lines.append("Invoice details:")
                    lines.extend(result["lines"])
                else:
                    lines = [f"💰 Bills (converted to {BASE_CURRENCY} for fair comparison)\n"] + result["lines"]
                    lines.append(f"\n{len(top_results)} result(s) shown.")

                answer = "\n".join(lines)

                set_last_results(session_id, top_results)
                update_active_state(session_id, {
                    "intent": "biggest_bills_converted",
                    "active_dataset": "invoice",
                    "result_count": len(top_results),
                    "last_sql": raw_sql
                })
                save_turn(session_id, "assistant", answer, {
                    "intent": "biggest_bills_converted",
                    "count": len(top_results),
                    "results_sample": top_results[:3]
                })
                return {
                    "answer": answer,
                    "intent": "biggest_bills_converted",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": raw_sql,
                    "results": top_results,
                    "count": len(top_results)
                }
            except Exception as e:
                logger.error(f"[{session_id}] Biggest bills conversion failed: {e}", exc_info=True)

    # ── Step 3j: Top-N + related docs — LLM extracts params, Python executes ─
    q_lower = question.lower()  # define locally for Step 3j
    has_ranking_signal = any(k in q_lower for k in [
        "top", "bottom", "highest", "lowest", "most", "least",
        "biggest", "smallest", "best", "worst", "first", "last"
    ])
    has_cross_signal = any(k in q_lower for k in [
        "and", "with", "plus", "also", "their", "show", "including",
        "along", "together", "as well", "too", "additionally"
    ])
    has_two_doc_types = sum([
        any(k in q_lower for k in [
            "invoice", "bill", "rechnung",
            "spending", "spent", "paid", "payment", "vendor spending",
            "EUR spending", "USD spending", "total paid", "amount paid"
        ]),
        any(k in q_lower for k in ["contract", "agreement", "vertrag"]),
        any(k in q_lower for k in ["receipt", "kassenbon"]),
    ]) >= 2

    logger.info(f"[{session_id}] Step 3j check: ranking={has_ranking_signal}, cross={has_cross_signal}, two_docs={has_two_doc_types}")
    cross_doc_intent_prompt = ChatPromptTemplate.from_template("""
You are a query intent classifier for a document analytics system.

Detect if this question asks for: ranked/ordered N items from one document type, PLUS related documents from another type.

Examples that ARE this pattern:
- "Compare top 3 vendors by EUR spending and show their contract values" → YES, top, invoice, contract
- "Top 5 vendors by invoice amount with their contracts" → YES, top, invoice, contract
- "Bottom 3 vendors by spending and their receipts" → YES, bottom, invoice, receipt
- "Show me the 5 biggest spenders in USD along with any agreements they have" → YES, top, invoice, contract
- "Who spends least in GBP? Show their invoices too" → YES, bottom, invoice, invoice
- "Last 3 vendors by contract value and their invoices" → YES, bottom, contract, invoice
- "Highest 2 vendors by total paid — show contracts" → YES, top, invoice, contract
- "Lowest spending vendors (top 4) with related contracts" → YES, bottom, invoice, contract
- "Give me 3 vendors with most receipts and their invoices" → YES, top, receipt, invoice
- "Zeige mir die 3 größten EUR-Lieferanten und ihre Verträge" → YES, top, invoice, contract

Examples that are NOT this pattern:
- "Show all invoices" → NO
- "Compare BrightPath vs FinEdge" → NO
- "Top vendors by spending" → NO (no secondary doc requested)
- "Show contracts" → NO

Question: {question}

If YES, return JSON:
{{
  "is_cross_doc_ranking": true,
  "n": <number, default 3>,
  "primary_doc": "invoice|contract|receipt",
  "secondary_doc": "invoice|contract|receipt",
  "currency": "EUR|USD|GBP|JPY|CHF|INR|null",
  "direction": "top|bottom",
  "metric": "spending|count|value"
}}

If NO, return:
{{"is_cross_doc_ranking": false}}

Return ONLY valid JSON. No explanation. No markdown.
""")

    if has_ranking_signal and has_cross_signal and has_two_doc_types:
        try:
            params_str = invoke_with_fallback(
                lambda llm: cross_doc_intent_prompt | llm,
                {"question": question}
            )
            # Clean JSON
            if "```" in params_str:
                params_str = params_str.split("```")[1]
                if params_str.startswith("json"):
                    params_str = params_str[4:]
            params = json.loads(params_str.strip())

            if params.get("is_cross_doc_ranking"):
                n = int(params.get("n", 3))
                primary_doc = params.get("primary_doc", "invoice")
                secondary_doc = params.get("secondary_doc", "contract")
                currency = params.get("currency")
                direction = params.get("direction", "top")
                order = "DESC" if direction == "top" else "ASC"

                currency_filter = f"AND r.extracted_data->>'currency' = '{currency}'" if currency and currency != "null" else ""
                metric_field = "total_amount" if primary_doc in ("invoice", "receipt") else "value"

                ranking_sql = f"""
                    SELECT
                        r.extracted_data->>'vendor_name' AS vendor,
                        ROUND(SUM(NULLIF(REGEXP_REPLACE(
                            r.extracted_data->>'{metric_field}',
                            '[^0-9.]', '', 'g'), '')::numeric), 2) AS total_value
                    FROM documents d
                    JOIN extraction_results r ON r.doc_id = d.id
                    WHERE r.document_type = '{primary_doc}'
                    AND d.status = 'done'
                    AND r.extracted_data->>'vendor_name' IS NOT NULL
                    {currency_filter}
                    GROUP BY vendor
                    ORDER BY total_value {order}
                    LIMIT {n}
                """

                is_valid, _ = validate_sql(ranking_sql)
                if is_valid:
                    ranking_results = await execute_query(ranking_sql, db)
                    top_vendors = [r["vendor"] for r in ranking_results if r.get("vendor")]

                    secondary_results = {}
                    for vendor in top_vendors:
                        safe_vendor = vendor.replace("'", "''")
                        if secondary_doc == "contract":
                            sec_sql = f"""
                                SELECT DISTINCT ON (d.filename)
                                    d.filename,
                                    r.extracted_data->>'parties' as parties,
                                    r.extracted_data->>'value' as value,
                                    r.extracted_data->>'currency' as currency
                                FROM documents d
                                JOIN extraction_results r ON r.doc_id = d.id
                                WHERE r.document_type = 'contract'
                                AND d.status = 'done'
                                AND r.extracted_data->>'parties' ILIKE '%{safe_vendor}%'
                                ORDER BY d.filename, d.created_at DESC
                                LIMIT 5
                            """
                        else:
                            sec_sql = f"""
                                SELECT DISTINCT ON (d.filename)
                                    d.filename,
                                    r.extracted_data->>'vendor_name' as vendor,
                                    r.extracted_data->>'total_amount' as amount,
                                    r.extracted_data->>'currency' as currency
                                FROM documents d
                                JOIN extraction_results r ON r.doc_id = d.id
                                WHERE r.document_type = '{secondary_doc}'
                                AND d.status = 'done'
                                AND r.extracted_data->>'vendor_name' ILIKE '%{safe_vendor}%'
                                ORDER BY d.filename, d.created_at DESC
                                LIMIT 5
                            """
                        is_valid_sec, _ = validate_sql(sec_sql)
                        if is_valid_sec:
                            secondary_results[vendor] = await execute_query(sec_sql, db)

                    # Build answer
                    currency_label = f" {currency}" if currency and currency != "null" else ""
                    lines = [f"🏆 {direction.title()} {n} vendors by {primary_doc} spending{currency_label}:\n"]

                    for i, row in enumerate(ranking_results, 1):
                        vendor = row.get("vendor", "Unknown")
                        total = row.get("total_value", 0)
                        try:
                            total_fmt = f"{float(total):,.2f}"
                        except (ValueError, TypeError):
                            total_fmt = str(total)
                        lines.append(f"{i}. {vendor}: {total_fmt}{currency_label}")

                        sec_docs = secondary_results.get(vendor, [])
                        if sec_docs:
                            lines.append(f"   📄 {secondary_doc.title()}s:")
                            for doc in sec_docs[:3]:
                                filename = doc.get("filename", "")
                                if secondary_doc == "contract":
                                    raw = doc.get("parties", "")
                                    try:
                                        import json as _json
                                        parsed = _json.loads(raw)
                                        if isinstance(parsed, list):
                                            names = [
                                                p.get("name", "") if isinstance(p, dict) else str(p)
                                                for p in parsed
                                                if (p.get("name", "") if isinstance(p, dict) else str(p))
                                                not in ("[MISSING_COMPANY_NAME]", "PARTY B (CLIENT)", "")
                                            ]
                                            parties_str = " / ".join(names)
                                        else:
                                            parties_str = str(raw)
                                    except Exception:
                                        parties_str = str(raw)
                                    val = doc.get("value", "N/A")
                                    cur = doc.get("currency", "")
                                    lines.append(f"   • {filename} | {parties_str} | {val} {cur}".strip())
                                else:
                                    amt = doc.get("amount", doc.get("total", "N/A"))
                                    cur = doc.get("currency", "")
                                    lines.append(f"   • {filename} | {amt} {cur}".strip())
                        else:
                            lines.append(f"   📄 No {secondary_doc}s found for {vendor}")
                        lines.append("")

                    answer = "\n".join(lines).strip()
                    set_last_results(session_id, ranking_results)
                    update_active_state(session_id, {
                        "intent": "cross_doc_ranking",
                        "active_dataset": primary_doc,
                        "result_count": len(ranking_results),
                        "last_sql": ranking_sql
                    })
                    save_turn(session_id, "assistant", answer, {
                        "intent": "cross_doc_ranking",
                        "count": len(ranking_results)
                    })
                    return {
                        "answer": answer,
                        "intent": "cross_doc_ranking",
                        "session_id": session_id,
                        "rewritten": None,
                        "sql": ranking_sql,
                        "results": ranking_results,
                        "count": len(ranking_results)
                    }
        except Exception as e:
            logger.error(f"[{session_id}] Cross-doc ranking failed: {e}", exc_info=True)
            try:
                await db.rollback()
            except Exception:
                pass
            # Fall through to normal SQL generation

    # ── Step 3x: "Something interesting" — auto-generated insights ──
    insight_keywords = [
        "something interesting", "any insights", "show me insights",
        "interesting about", "tell me something interesting",
        "surprise me", "what's interesting", "give me insights"
    ]
    if special_intent == "something_interesting":
        logger.info(f"[{session_id}] Auto-insights triggered")
        try:
            overview_sql = """
                SELECT
                    COUNT(*) as invoice_count,
                    COUNT(DISTINCT r.extracted_data->>'vendor_name') as vendor_count,
                    COUNT(DISTINCT r.extracted_data->>'currency') as currency_count
                FROM documents d
                JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type = 'invoice' AND d.status = 'done'
            """
            overview = (await execute_query(overview_sql, db))[0]

            currency_sql = """
                SELECT r.extracted_data->>'currency' as currency, COUNT(*) as count
                FROM documents d
                JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type = 'invoice' AND d.status = 'done'
                AND r.extracted_data->>'currency' IS NOT NULL
                GROUP BY currency ORDER BY count DESC
            """
            currency_rows = await execute_query(currency_sql, db)

            raw_sql = """
                SELECT
                    r.extracted_data->>'vendor_name' as vendor,
                    NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric as amount,
                    r.extracted_data->>'currency' as currency
                FROM documents d
                JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type = 'invoice' AND d.status = 'done'
                AND r.extracted_data->>'total_amount' IS NOT NULL
                AND r.extracted_data->>'vendor_name' IS NOT NULL
            """
            raw_rows = await execute_query(raw_sql, db)
            rates = await get_exchange_rates()

            vendor_totals = {}
            largest_invoice = None
            total_converted = 0.0
            for row in raw_rows:
                amt, cur, vendor = row.get("amount"), row.get("currency"), row.get("vendor")
                if amt is None or not cur:
                    continue
                base_amt = convert_to_base(float(amt), cur, rates)
                if base_amt is None:
                    continue
                total_converted += base_amt
                vendor_totals[vendor] = vendor_totals.get(vendor, 0) + base_amt
                if largest_invoice is None or base_amt > largest_invoice["base_amount"]:
                    largest_invoice = {**row, "base_amount": base_amt}

            top_vendor = max(vendor_totals.items(), key=lambda x: x[1]) if vendor_totals else None

            dup_sql = """
                SELECT COUNT(*) as dup_count FROM (
                    SELECT r.extracted_data->>'invoice_number' as inv
                    FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                    WHERE d.status='done' AND r.document_type='invoice'
                    AND r.extracted_data->>'invoice_number' IS NOT NULL
                    AND r.extracted_data->>'invoice_number' != ''
                    GROUP BY r.extracted_data->>'invoice_number',
                             r.extracted_data->>'vendor_name',
                             r.extracted_data->>'total_amount'
                    HAVING COUNT(*) > 1
                ) sub
            """
            dup_result = await execute_query(dup_sql, db)
            dup_count = dup_result[0].get("dup_count", 0) if dup_result else 0

            overdue_sql = r"""
                SELECT COUNT(*) as overdue_count
                FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type='invoice' AND d.status='done'
                AND r.extracted_data->>'due_date' IS NOT NULL
                AND r.extracted_data->>'due_date' != ''
                AND (
                    CASE
                        WHEN r.extracted_data->>'due_date' ~ '^\d{4}-\d{2}-\d{2}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'YYYY-MM-DD')
                        WHEN r.extracted_data->>'due_date' ~ '^\d{1,2}/\d{1,2}/\d{4}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'DD/MM/YYYY')
                        WHEN r.extracted_data->>'due_date' ~ '^\d{1,2} \w+ \d{4}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY')
                        ELSE NULL
                    END
                ) < CURRENT_DATE
            """
            overdue_result = await execute_query(overdue_sql, db)
            overdue_count = overdue_result[0].get("overdue_count", 0) if overdue_result else 0

            lines = ["💡 Here's something interesting about your invoices:\n"]
            invoice_count = overview.get("invoice_count", 0)
            vendor_count = overview.get("vendor_count", 0)
            currency_count = overview.get("currency_count", 0)
            lines.append(f"You have {invoice_count} invoices from {vendor_count} vendors across {currency_count} currencies.")

            if top_vendor:
                vendor_name, vendor_total = top_vendor
                share = (vendor_total / total_converted * 100) if total_converted else 0
                lines.append(
                    f"{vendor_name} is your top vendor by converted spend "
                    f"(≈{vendor_total:,.2f} EUR — {share:.1f}% of your total)."
                )

            if currency_rows:
                dominant = currency_rows[0]
                dom_share = (dominant['count'] / invoice_count * 100) if invoice_count else 0
                lines.append(
                    f"{dominant['currency']} is your most common currency "
                    f"({dominant['count']} invoices, {dom_share:.1f}% of all)."
                )

            if largest_invoice:
                lines.append(
                    f"Your single largest invoice is {largest_invoice['vendor']} at "
                    f"{float(largest_invoice['amount']):,.2f} {largest_invoice['currency']} "
                    f"(≈{largest_invoice['base_amount']:,.2f} EUR)."
                )

            if dup_count > 0:
                lines.append(f"⚠️ {dup_count} invoice group(s) appear to be duplicates — worth reviewing.")
            if overdue_count > 0:
                lines.append(f"⚠️ {overdue_count} invoice(s) are currently overdue.")

            answer = "\n".join(lines)
            save_turn(session_id, "assistant", answer, {"intent": "auto_insights"})
            return {
                "answer": answer, "intent": "auto_insights", "session_id": session_id,
                "rewritten": None, "sql": None, "results": [], "count": 0
            }
        except Exception as e:
            logger.error(f"[{session_id}] Auto-insights failed: {e}", exc_info=True)

    # ── Step 3y: "Missing data" — data quality audit ──
    missing_data_keywords = [
        "missing data", "any missing", "incomplete data", "data quality",
        "missing information", "missing fields", "what's missing",
        "any incomplete", "missing values"
    ]
    if special_intent == "missing_data":
        logger.info(f"[{session_id}] Missing data audit triggered")
        try:
            invoice_sql = """
                SELECT
                    COUNT(*) FILTER (WHERE r.extracted_data->>'vendor_name' IS NULL OR r.extracted_data->>'vendor_name' = '') as missing_vendor,
                    COUNT(*) FILTER (WHERE r.extracted_data->>'total_amount' IS NULL OR r.extracted_data->>'total_amount' = '') as missing_amount,
                    COUNT(*) FILTER (WHERE r.extracted_data->>'currency' IS NULL OR r.extracted_data->>'currency' = '') as missing_currency,
                    COUNT(*) FILTER (WHERE r.extracted_data->>'due_date' IS NULL OR r.extracted_data->>'due_date' = '') as missing_due_date,
                    COUNT(*) as total_docs
                FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                WHERE d.status = 'done' AND r.document_type = 'invoice'
            """
            contract_sql = """
                SELECT
                    COUNT(*) FILTER (WHERE r.extracted_data->>'parties' IS NULL OR r.extracted_data->>'parties' = '') as missing_parties,
                    COUNT(*) FILTER (WHERE r.extracted_data->>'value' IS NULL OR r.extracted_data->>'value' = '') as missing_value,
                    COUNT(*) FILTER (WHERE r.extracted_data->>'end_date' IS NULL OR r.extracted_data->>'end_date' = '') as missing_end_date,
                    COUNT(*) as total_docs
                FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                WHERE d.status = 'done' AND r.document_type = 'contract'
            """
            receipt_sql = """
                SELECT
                    COUNT(*) FILTER (WHERE r.extracted_data->>'vendor_name' IS NULL OR r.extracted_data->>'vendor_name' = '') as missing_vendor,
                    COUNT(*) FILTER (WHERE r.extracted_data->>'total_amount' IS NULL OR r.extracted_data->>'total_amount' = '') as missing_amount,
                    COUNT(*) as total_docs
                FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                WHERE d.status = 'done' AND r.document_type = 'receipt'
            """

            invoice_stats = (await execute_query(invoice_sql, db))[0]
            contract_stats = (await execute_query(contract_sql, db))[0]
            receipt_stats = (await execute_query(receipt_sql, db))[0]

            findings = []
            inv_total = invoice_stats.get("total_docs", 0)
            if inv_total:
                if invoice_stats.get("missing_vendor", 0) > 0:
                    findings.append(f"{invoice_stats['missing_vendor']} invoice(s) missing vendor name")
                if invoice_stats.get("missing_amount", 0) > 0:
                    findings.append(f"{invoice_stats['missing_amount']} invoice(s) missing total amount")
                if invoice_stats.get("missing_currency", 0) > 0:
                    findings.append(f"{invoice_stats['missing_currency']} invoice(s) missing currency")
                if invoice_stats.get("missing_due_date", 0) > 0:
                    findings.append(f"{invoice_stats['missing_due_date']} invoice(s) missing due date")

            con_total = contract_stats.get("total_docs", 0)
            if con_total:
                if contract_stats.get("missing_parties", 0) > 0:
                    findings.append(f"{contract_stats['missing_parties']} contract(s) missing parties")
                if contract_stats.get("missing_value", 0) > 0:
                    findings.append(f"{contract_stats['missing_value']} contract(s) missing value")
                if contract_stats.get("missing_end_date", 0) > 0:
                    findings.append(f"{contract_stats['missing_end_date']} contract(s) missing end date")

            rec_total = receipt_stats.get("total_docs", 0)
            if rec_total:
                if receipt_stats.get("missing_vendor", 0) > 0:
                    findings.append(f"{receipt_stats['missing_vendor']} receipt(s) missing vendor name")
                if receipt_stats.get("missing_amount", 0) > 0:
                    findings.append(f"{receipt_stats['missing_amount']} receipt(s) missing total amount")

            total_docs = inv_total + con_total + rec_total

            if not findings:
                answer = f"✅ No missing data found across {total_docs} documents. All critical fields are populated."
            else:
                lines = [f"📋 Data Quality Check — across {total_docs} documents:\n"]
                for f in findings:
                    lines.append(f"⚠️ {f}")
                answer = "\n".join(lines)

            save_turn(session_id, "assistant", answer, {"intent": "missing_data_audit"})
            return {
                "answer": answer, "intent": "missing_data_audit", "session_id": session_id,
                "rewritten": None, "sql": None, "results": [], "count": 0
            }
        except Exception as e:
            logger.error(f"[{session_id}] Missing data audit failed: {e}", exc_info=True)        

    # ── Step 3z: "Which vendors should I worry about" — risk assessment ──
    risk_keywords = [
        "worried about", "be worried", "concerned about", "should i be concerned",
        "risky vendor", "risky vendors", "any risk", "any concerns",
        "vendors to watch", "problematic vendor", "vendor risk"
    ]
    if special_intent == "vendor_risk":
        logger.info(f"[{session_id}] Vendor risk assessment triggered")
        try:
            dup_sql = """
                SELECT vendor_name as vendor, inv as invoice_number, amt as amount, COUNT(*) as occurrences
                FROM (
                    SELECT
                        r.extracted_data->>'vendor_name' as vendor_name,
                        r.extracted_data->>'invoice_number' as inv,
                        r.extracted_data->>'total_amount' as amt
                    FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                    WHERE d.status='done' AND r.document_type='invoice'
                    AND r.extracted_data->>'invoice_number' IS NOT NULL
                    AND r.extracted_data->>'invoice_number' != ''
                ) dup
                GROUP BY vendor_name, inv, amt
                HAVING COUNT(*) > 1
                ORDER BY vendor_name
            """
            overdue_sql = r"""
                SELECT r.extracted_data->>'vendor_name' as vendor,
                       d.filename,
                       r.extracted_data->>'due_date' as due_date,
                       r.extracted_data->>'total_amount' as amount,
                       r.extracted_data->>'currency' as currency
                FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type='invoice' AND d.status='done'
                AND r.extracted_data->>'due_date' IS NOT NULL
                AND r.extracted_data->>'due_date' != ''
                AND (
                    CASE
                        WHEN r.extracted_data->>'due_date' ~ '^\d{4}-\d{2}-\d{2}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'YYYY-MM-DD')
                        WHEN r.extracted_data->>'due_date' ~ '^\d{1,2}/\d{1,2}/\d{4}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'DD/MM/YYYY')
                        WHEN r.extracted_data->>'due_date' ~ '^\d{1,2}\.\d{1,2}\.\d{4}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'DD.MM.YYYY')
                        WHEN r.extracted_data->>'due_date' ~ '^\d{1,2} \w+ \d{4}$'
                            THEN TO_DATE(r.extracted_data->>'due_date', 'DD Month YYYY')
                        ELSE NULL
                    END
                ) < CURRENT_DATE
                AND r.extracted_data->>'vendor_name' IS NOT NULL
                ORDER BY vendor
            """

            dup_rows = await execute_query(dup_sql, db)
            overdue_rows = await execute_query(overdue_sql, db)

            all_invoices_sql = """
                SELECT d.filename, r.extracted_data->>'vendor_name' as vendor,
                       NULLIF(REGEXP_REPLACE(r.extracted_data->>'total_amount','[^0-9.]','','g'),'')::numeric as amount,
                       r.extracted_data->>'currency' as currency
                FROM documents d JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type='invoice' AND d.status='done'
                AND r.extracted_data->>'total_amount' IS NOT NULL
            """
            all_invoices = await execute_query(all_invoices_sql, db)
            outliers = detect_outliers(all_invoices)
            outlier_by_vendor = {}
            for o in outliers:
                v = o.get("vendor")
                if v:
                    outlier_by_vendor.setdefault(v, []).append(o)

            risk_scores = {}
            reasons = {}

            # ── Duplicates: group by vendor, list invoice numbers ──
            dup_by_vendor = {}
            for row in dup_rows:
                v = row.get("vendor")
                if not v:
                    continue
                dup_by_vendor.setdefault(v, []).append(row)

            for v, rows in dup_by_vendor.items():
                risk_scores[v] = risk_scores.get(v, 0) + len(rows) * 2
                parts = [
                    f"{r.get('invoice_number')} ({r.get('occurrences')}x, {r.get('amount')})"
                    for r in rows[:3]
                ]
                more = f", +{len(rows)-3} more" if len(rows) > 3 else ""
                reasons.setdefault(v, []).append(
                    f"{len(rows)} duplicate invoice group(s): {'; '.join(parts)}{more}"
                )

            # ── Overdue: group by vendor, list filename/amount/due date ──
            overdue_by_vendor = {}
            for row in overdue_rows:
                v = row.get("vendor")
                if not v:
                    continue
                overdue_by_vendor.setdefault(v, []).append(row)

            for v, rows in overdue_by_vendor.items():
                risk_scores[v] = risk_scores.get(v, 0) + len(rows)
                parts = [
                    f"{r.get('filename')} ({r.get('amount')} {r.get('currency')}, due {r.get('due_date')})"
                    for r in rows[:3]
                ]
                more = f", +{len(rows)-3} more" if len(rows) > 3 else ""
                reasons.setdefault(v, []).append(
                    f"{len(rows)} overdue invoice(s): {'; '.join(parts)}{more}"
                )

            # ── Outliers: group by vendor, list filename/amount ──
            for v, rows in outlier_by_vendor.items():
                risk_scores[v] = risk_scores.get(v, 0) + len(rows) * 1.5
                parts = [
                    f"{r.get('filename')} ({r.get('amount')} {r.get('currency')})"
                    for r in rows[:3]
                ]
                more = f", +{len(rows)-3} more" if len(rows) > 3 else ""
                reasons.setdefault(v, []).append(
                    f"{len(rows)} unusually large invoice(s): {'; '.join(parts)}{more}"
                )

            if not risk_scores:
                answer = "✅ No vendors currently show signs of risk — no duplicates, overdue invoices, or unusual amounts detected."
            else:
                ranked_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
                selection = parse_ranking_selection(question)

                def format_risk_line(vendor, score, idx=None):
                    prefix = f"{idx}. " if idx else "• "
                    why = ", ".join(reasons.get(vendor, ["no specific risk signals"]))
                    return f"{prefix}{vendor} — risk score {score:.1f} ({why})"

                result = select_from_ranking(ranked_risk, selection, format_risk_line)
                lines = ["⚠️ Vendor Risk Assessment:\n"] + result["lines"]
                answer = "\n".join(lines)

            save_turn(session_id, "assistant", answer, {"intent": "vendor_risk_assessment"})
            return {
                "answer": answer, "intent": "vendor_risk_assessment", "session_id": session_id,
                "rewritten": None, "sql": None, "results": [], "count": 0
            }
        except Exception as e:
            logger.error(f"[{session_id}] Vendor risk assessment failed: {e}", exc_info=True)

    # ── Step 3w: "Is spending increasing or decreasing" — true trend with
    # currency conversion AND full transparency on excluded rows ──
    if special_intent == "spending_trend":
        logger.info(f"[{session_id}] Spending direction trend triggered")
        try:
            raw_sql = """
                SELECT
                    r.extracted_data->>'issue_date' as issue_date,
                    r.extracted_data->>'total_amount' as raw_amount,
                    r.extracted_data->>'currency' as currency
                FROM documents d
                JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type = 'invoice' AND d.status = 'done'
            """
            raw_rows = await execute_query(raw_sql, db)
            rates = await get_exchange_rates()
            rates_unavailable = len(rates) <= 1  # only has BASE_CURRENCY itself, or truly empty

            monthly_totals = {}
            monthly_counts = {}
            total_processed = 0
            excluded_no_date = 0
            excluded_no_amount = 0
            excluded_no_currency = 0
            excluded_unknown_currency = 0

            for row in raw_rows:
                total_processed += 1
                amt = clean_amount_str(row.get("raw_amount"))
                currency = row.get("currency")

                if amt is None:
                    excluded_no_amount += 1
                    continue
                if not currency:
                    excluded_no_currency += 1
                    continue
                dt = parse_mixed_date(row.get("issue_date"))
                if not dt:
                    excluded_no_date += 1
                    continue
                base_amt = convert_to_base(amt, currency, rates)
                if base_amt is None:
                    excluded_unknown_currency += 1
                    continue

                key = dt.strftime("%Y-%m")
                monthly_totals[key] = monthly_totals.get(key, 0) + base_amt
                monthly_counts[key] = monthly_counts.get(key, 0) + 1

            # ── One LLM call decides selection AND metric semantically —
            # no keyword lists anywhere in this handler ──
            selection = parse_ranking_selection(question)
            wants_specific_month = selection.get("selection_type") in (
                "single_rank", "top_n", "bottom_n", "tier", "average", "compare"
            )
            wants_count_metric = selection.get("metric") == "count"

            if wants_specific_month and monthly_totals:
                metric_map = monthly_counts if wants_count_metric else monthly_totals
                ranked_months = sorted(
                    [(_dt.strptime(k, "%Y-%m").strftime("%b %Y"), v) for k, v in metric_map.items()],
                    key=lambda x: x[1], reverse=True
                )

                def format_month_line(label, value, idx=None):
                    count = next((monthly_counts[k] for k in monthly_totals
                                  if _dt.strptime(k, "%Y-%m").strftime("%b %Y") == label), 0)
                    total = next((monthly_totals[k] for k in monthly_totals
                                  if _dt.strptime(k, "%Y-%m").strftime("%b %Y") == label), 0)
                    prefix = f"{idx}. " if idx else "📈 "
                    return f"{prefix}{label}: {count} invoices | {total:,.2f} {BASE_CURRENCY}"

                result = select_from_ranking(ranked_months, selection, format_month_line)
                header = f"📊 Monthly {'Invoice Count' if wants_count_metric else 'Spending'} Ranking"
                if not wants_count_metric:
                    header += f" (converted to {BASE_CURRENCY})"
                lines = [header + ":\n"] + result["lines"]

                if rates_unavailable:
                    lines.append(f"\n⚠️ Exchange rate service unavailable — results may only include {BASE_CURRENCY}-denominated invoices and could be incomplete.")

                answer = "\n".join(lines)
                save_turn(session_id, "assistant", answer, {"intent": "spending_trend_converted"})
                return {
                    "answer": answer, "intent": "spending_trend_converted", "session_id": session_id,
                    "rewritten": None, "sql": None, "results": [], "count": 0
                }

            # ── Otherwise show the full trend ──
            sorted_months = sorted(monthly_totals.keys())
            lines = [f"📈 Monthly Spending Trend (converted to {BASE_CURRENCY})\n"]
            for key in sorted_months:
                label = _dt.strptime(key, "%Y-%m").strftime("%b %Y")
                lines.append(f"{label}: {monthly_counts[key]} invoices | {monthly_totals[key]:,.2f} {BASE_CURRENCY}")

            if len(sorted_months) >= 2:
                half = len(sorted_months) // 2
                early_avg = sum(monthly_totals[m] for m in sorted_months[:half]) / max(half, 1)
                recent_avg = sum(monthly_totals[m] for m in sorted_months[half:]) / max(len(sorted_months) - half, 1)
                if recent_avg > early_avg * 1.05:
                    insight = f"📊 Spending is increasing — recent average ({recent_avg:,.2f} {BASE_CURRENCY}) is higher than earlier average ({early_avg:,.2f} {BASE_CURRENCY})."
                elif recent_avg < early_avg * 0.95:
                    insight = f"📊 Spending is decreasing — recent average ({recent_avg:,.2f} {BASE_CURRENCY}) is lower than earlier average ({early_avg:,.2f} {BASE_CURRENCY})."
                else:
                    insight = "📊 Spending has remained relatively stable."
                lines.append(f"\n{insight}")

            lines.append(f"\n{len(sorted_months)} months analyzed.")

            total_excluded = excluded_no_date + excluded_no_amount + excluded_no_currency + excluded_unknown_currency
            if total_excluded > 0:
                parts = []
                if excluded_no_date:
                    parts.append(f"{excluded_no_date} with unparseable issue date")
                if excluded_no_amount:
                    parts.append(f"{excluded_no_amount} with missing/invalid amount")
                if excluded_no_currency:
                    parts.append(f"{excluded_no_currency} with missing currency")
                if excluded_unknown_currency:
                    parts.append(f"{excluded_unknown_currency} with unsupported currency")
                lines.append(
                    f"⚠️ {total_excluded} of {total_processed} invoice(s) excluded from this trend "
                    f"({', '.join(parts)})."
                )

            answer = "\n".join(lines)
            save_turn(session_id, "assistant", answer, {"intent": "spending_trend_converted"})
            return {
                "answer": answer, "intent": "spending_trend_converted", "session_id": session_id,
                "rewritten": None, "sql": None, "results": [], "count": 0
            }
        except Exception as e:
            logger.error(f"[{session_id}] Spending direction trend failed: {e}", exc_info=True)

            # ── Otherwise show the full trend (existing code continues below) ──
            sorted_months = sorted(monthly_totals.keys())
            lines = [f"📈 Monthly Spending Trend (converted to {BASE_CURRENCY})\n"]
            for key in sorted_months:
                label = _dt.strptime(key, "%Y-%m").strftime("%b %Y")
                lines.append(f"{label}: {monthly_counts[key]} invoices | {monthly_totals[key]:,.2f} {BASE_CURRENCY}")

            if len(sorted_months) >= 2:
                half = len(sorted_months) // 2
                early_avg = sum(monthly_totals[m] for m in sorted_months[:half]) / max(half, 1)
                recent_avg = sum(monthly_totals[m] for m in sorted_months[half:]) / max(len(sorted_months) - half, 1)
                if recent_avg > early_avg * 1.05:
                    insight = f"📊 Spending is increasing — recent average ({recent_avg:,.2f} {BASE_CURRENCY}) is higher than earlier average ({early_avg:,.2f} {BASE_CURRENCY})."
                elif recent_avg < early_avg * 0.95:
                    insight = f"📊 Spending is decreasing — recent average ({recent_avg:,.2f} {BASE_CURRENCY}) is lower than earlier average ({early_avg:,.2f} {BASE_CURRENCY})."
                else:
                    insight = "📊 Spending has remained relatively stable."
                lines.append(f"\n{insight}")

            lines.append(f"\n{len(sorted_months)} months analyzed.")

            total_excluded = excluded_no_date + excluded_no_amount + excluded_no_currency + excluded_unknown_currency
            if total_excluded > 0:
                parts = []
                if excluded_no_date:
                    parts.append(f"{excluded_no_date} with unparseable issue date")
                if excluded_no_amount:
                    parts.append(f"{excluded_no_amount} with missing/invalid amount")
                if excluded_no_currency:
                    parts.append(f"{excluded_no_currency} with missing currency")
                if excluded_unknown_currency:
                    parts.append(f"{excluded_unknown_currency} with unsupported currency")
                lines.append(
                    f"⚠️ {total_excluded} of {total_processed} invoice(s) excluded from this trend "
                    f"({', '.join(parts)})."
                )

            answer = "\n".join(lines)
            save_turn(session_id, "assistant", answer, {"intent": "spending_trend_converted"})
            return {
                "answer": answer, "intent": "spending_trend_converted", "session_id": session_id,
                "rewritten": None, "sql": None, "results": [], "count": 0
            }
        except Exception as e:
            logger.error(f"[{session_id}] Spending direction trend failed: {e}", exc_info=True)

    # ── Step 3u: "Most important vendor" — composite score
    # (converted spend + frequency + contract presence), fully transparent ──
    if special_intent == "most_important_vendor":
        logger.info(f"[{session_id}] Most important vendor analysis triggered")
        try:
            raw_sql = """
                SELECT
                    r.extracted_data->>'vendor_name' as vendor,
                    r.extracted_data->>'total_amount' as raw_amount,
                    r.extracted_data->>'currency' as currency
                FROM documents d
                JOIN extraction_results r ON r.doc_id = d.id
                WHERE r.document_type = 'invoice' AND d.status = 'done'
                AND r.extracted_data->>'vendor_name' IS NOT NULL
            """
            raw_rows = await execute_query(raw_sql, db)
            rates = await get_exchange_rates()

            import re as _re

            def clean_amount(raw):
                if raw is None:
                    return None
                cleaned = _re.sub(r'[^0-9.]', '', str(raw))
                if not cleaned:
                    return None
                try:
                    return float(cleaned)
                except ValueError:
                    return None

            vendor_spend = {}
            vendor_freq = {}
            vendor_display_name = {}  # normalized key -> cleanest/shortest display label
            excluded_count = 0

            for row in raw_rows:
                vendor = row.get("vendor")
                if not vendor:
                    continue
                norm = normalize(vendor) or vendor.lower()

                # Keep the shortest/cleanest-looking variant as the display name
                if norm not in vendor_display_name or len(vendor) < len(vendor_display_name[norm]):
                    vendor_display_name[norm] = vendor

                amt = clean_amount(row.get("raw_amount"))
                currency = row.get("currency")
                vendor_freq[norm] = vendor_freq.get(norm, 0) + 1
                if amt is None or not currency:
                    excluded_count += 1
                    continue
                base_amt = convert_to_base(amt, currency, rates)
                if base_amt is None:
                    excluded_count += 1
                    continue
                vendor_spend[norm] = vendor_spend.get(norm, 0) + base_amt

            contract_sql = """
                SELECT DISTINCT r2.extracted_data->>'parties' as parties
                FROM documents d2 JOIN extraction_results r2 ON r2.doc_id = d2.id
                WHERE r2.document_type = 'contract' AND d2.status = 'done'
                AND r2.extracted_data->>'parties' IS NOT NULL
            """
            contract_rows = await execute_query(contract_sql, db)
            all_parties_text = " | ".join(
                (row.get("parties") or "").lower() for row in contract_rows
            )

            def has_contract(vendor_name: str) -> bool:
                return vendor_name.lower() in all_parties_text

            max_spend = max(vendor_spend.values()) if vendor_spend else 1
            max_freq = max(vendor_freq.values()) if vendor_freq else 1

            scores = {}
            details = {}
            all_vendors = set(vendor_spend.keys()) | set(vendor_freq.keys())
            for norm_key in all_vendors:
                v = vendor_display_name.get(norm_key, norm_key)
                spend = vendor_spend.get(norm_key, 0)
                freq = vendor_freq.get(norm_key, 0)
                contract = has_contract(v)
                spend_score = (spend / max_spend * 100) if max_spend else 0
                freq_score = (freq / max_freq * 100) if max_freq else 0
                contract_score = 100 if contract else 0
                composite = spend_score * 0.6 + freq_score * 0.25 + contract_score * 0.15
                scores[v] = composite
                details[v] = {
                    "spend": spend, "freq": freq, "contract": contract,
                    "spend_score": spend_score, "freq_score": freq_score
                }

            if not scores:
                answer = "No vendor data available to determine importance."
            else:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selection = parse_ranking_selection(question)

                def format_vendor_line(name, score, idx=None):
                    dd = details[name]
                    prefix = f"{idx}. " if idx else "🏆 "
                    return (
                        f"{prefix}{name} — {score:.1f}/100 "
                        f"(spend: {dd['spend']:,.2f} {BASE_CURRENCY}, {dd['freq']} invoices"
                        f"{', has contract' if dd['contract'] else ''})"
                    )

                is_default_top_query = (
                    selection.get("selection_type") == "single_rank"
                    and (selection.get("rank_position") or 1) == 1
                )

                if is_default_top_query:
                    top_vendor, top_score = ranked[0]
                    d = details[top_vendor]

                    lines = [f"🏆 Most important vendor: {top_vendor}\n"]
                    lines.append("Why:")
                    lines.append(
                        f"• Total spend: {d['spend']:,.2f} {BASE_CURRENCY} "
                        f"({d['spend_score']:.0f}% of your top spender's level)"
                    )
                    lines.append(f"• Frequency: {d['freq']} invoices ({d['freq_score']:.0f}% of most frequent vendor)")
                    lines.append(f"• Contract on file: {'Yes ✅' if d['contract'] else 'No'}")
                    lines.append(f"\nImportance score: {top_score:.1f}/100\n")

                    lines.append("Runner-ups:")
                    for v, s in ranked[1:4]:
                        dd = details[v]
                        lines.append(
                            f"• {v} — {s:.1f}/100 "
                            f"({dd['spend']:,.2f} {BASE_CURRENCY}, {dd['freq']} invoices"
                            f"{', has contract' if dd['contract'] else ''})"
                        )
                else:
                    result = select_from_ranking(ranked, selection, format_vendor_line)
                    lines = ["📊 Vendor Importance:\n"] + result["lines"]

                if excluded_count > 0:
                    lines.append(f"\n⚠️ {excluded_count} invoice(s) excluded from spend calculation (missing/invalid amount or currency).")

                answer = "\n".join(lines)

            save_turn(session_id, "assistant", answer, {"intent": "most_important_vendor"})
            return {
                "answer": answer, "intent": "most_important_vendor", "session_id": session_id,
                "rewritten": None, "sql": None, "results": [], "count": 0
            }
        except Exception as e:
            logger.error(f"[{session_id}] Most important vendor analysis failed: {e}", exc_info=True)

    # ── Step 4: Decompose if complex ─────────────
    if is_complex_question(question):
        logger.info(f"[{session_id}] Complex question — decomposing")
        sub_questions = decompose_question(question)
        sub_results = []
        resolved_entities = {}   # ← carries vendor name forward across sub-questions

        vague_vendor_patterns = [
            r"the vendor with (the )?highest spending( this year)?",
            r"the vendor with the most (total )?spending",
            r"the vendor with the highest total",
            r"the top vendor",
            r"\bthis vendor\b", r"\bthat vendor\b",
            r"\btheir\b", r"\bthey\b",
            r"\bthe vendor\b",
        ]

        for sub_q in sub_questions:
            try:
                # ── Inject resolved vendor from earlier sub-questions ──
                injected_q = sub_q
                if resolved_entities.get("vendor"):
                    vendor_name = resolved_entities["vendor"]
                    for pattern in vague_vendor_patterns:
                        injected_q = re.sub(pattern, vendor_name, injected_q, flags=re.IGNORECASE)
                    if injected_q != sub_q:
                        logger.info(f"[{session_id}] Injected vendor '{vendor_name}': '{sub_q}' -> '{injected_q}'")

                sub_context = resolve_context(
                    injected_q, history, last_metadata,
                    last_results=last_results,
                    conversation_focus=conversation_focus
                )
                sub_rewritten = rewrite_query(injected_q, history, last_metadata, sub_context)
                sub_sql = generate_sql(sub_rewritten, history, sub_context)

                if not sub_sql:
                    continue

                is_valid, reason = validate_sql(sub_sql)
                if not is_valid:
                    continue

                sub_data = await execute_query(sub_sql, db)

                # ── Capture vendor entity for next sub-questions ──
                if sub_data and not resolved_entities.get("vendor"):
                    first_row = sub_data[0]
                    vendor_val = first_row.get("vendor") or first_row.get("vendor_name")
                    if vendor_val:
                        resolved_entities["vendor"] = vendor_val
                        logger.info(f"[{session_id}] Resolved vendor entity: {vendor_val}")

                sub_results.append({
                    "question": injected_q,
                    "results": sub_data,
                    "count": len(sub_data)
                })
            except Exception as e:
                logger.error(f"[{session_id}] Sub-query failed: {e}", exc_info=True)
                continue

        if sub_results:
            from chatbot.response_synthesizer import synthesize_multi
            answer = synthesize_multi(question, sub_results, history)
            save_turn(session_id, "assistant", answer, {
                "intent": intent,
                "decomposed": True,
                "sub_questions": sub_questions
            })
            return {
                "answer": answer,
                "intent": intent,
                "session_id": session_id,
                "rewritten": None,
                "sql": None,
                "results": [],
                "count": sum(r["count"] for r in sub_results),
                "decomposed": True
            }

    # ── Step 5: Resolve context ───────────────────
    resolved_context = resolve_context(
        question, history, last_metadata,
        last_results=last_results,
        conversation_focus=conversation_focus
    )
    logger.info(f"[{session_id}] Context: {resolved_context}")

    # ── Step 5.5: Handle direct memory answers ────
    intent_override = resolved_context.get("intent_override")
    q_lower = question.lower()

    if intent_override == "repetition_count" and last_results:
        count = len(last_results)
        vendor = (
            last_results[0].get("vendor") or
            last_results[0].get("vendor_name") or
            conversation_focus.get("vendor", "")
        )
        answer = f"{vendor} appears {count} times." if vendor else f"It appears {count} times."
        save_turn(session_id, "assistant", answer, {"intent": "repetition_count"})
        return {
            "answer": answer,
            "intent": "repetition_count",
            "session_id": session_id,
            "rewritten": None,
            "sql": None,
            "results": last_results,
            "count": count
        }

    if intent_override == "list_navigation" and last_results:
        index = 0
        word_ordinals = {
            "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
            "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
        }
        matched_ordinal = next((v for k, v in word_ordinals.items() if k in q_lower), None)

        # Numeric ordinal: "100th", "23rd", "5th", "1st" etc.
        numeric_match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\b', question)

        if "last" in q_lower:
            index = -1
        elif matched_ordinal is not None:
            index = matched_ordinal
        elif numeric_match:
            index = int(numeric_match.group(1)) - 1

        if abs(index) < len(last_results):
            item = last_results[index]
            lines = [f"Item {index + 1 if index >= 0 else len(last_results)} from previous results:\n"]
            for k, v in item.items():
                if v is not None:
                    lines.append(f"• {k.replace('_', ' ').title()}: {v}")
            answer = "\n".join(lines)
            save_turn(session_id, "assistant", answer, {"intent": "list_navigation"})
            return {
                "answer": answer,
                "intent": "list_navigation",
                "session_id": session_id,
                "rewritten": None,
                "sql": None,
                "results": [item],
                "count": 1
            }
    # ── Universal check: if question mentions a DIFFERENT document type than
    # the active dataset, this is ALWAYS standalone — never a transform.
    # This replaces fragile keyword-list maintenance with direct type detection.
    active_dataset = active_state.get("active_dataset", "")
    doc_type_mentions = {
        "invoice": ["invoice", "bill", "rechnung"],
        "contract": ["contract", "agreement", "vertrag"],
        "receipt": ["receipt", "kassenbon"],
        "report": ["report"]
    }
    mentioned_types = [
        dtype for dtype, keywords in doc_type_mentions.items()
        if any(kw in question.lower() for kw in keywords)
    ]
    document_type_mismatch = bool(
        active_dataset and mentioned_types and active_dataset not in mentioned_types
    )
    if document_type_mismatch:
        logger.info(f"[{session_id}] Document type mismatch: active={active_dataset}, "
                    f"mentioned={mentioned_types} — forcing standalone")
                
    # ── Step 5.55: Universal "Nth item" position query ────────────────────────
    # Handles: "5th invoice", "12th contract", "2 receipts", "show me 75th",
    # "second invoice", "the third contract" — any number (digit or word),
    # any document type, with or without explicit type (inherits from
    # active_dataset when type is omitted).
    ORDINAL_WORDS = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
        "fifteenth": 15, "twentieth": 20, "thirtieth": 30,
    }
    DOC_TYPE_WORDS = {
        "invoice": "invoice", "invoices": "invoice", "bill": "invoice", "bills": "invoice",
        "contract": "contract", "contracts": "contract", "agreement": "contract", "agreements": "contract",
        "receipt": "receipt", "receipts": "receipt",
        "document": None, "documents": None,
    }

    def _ordinal_suffix(n: int) -> str:
        if 10 <= n % 100 <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    q_words = re.findall(r"[A-Za-z0-9']+", question.lower())

    requested_n = None
    doc_type_found = None

    for idx, w in enumerate(q_words):
        n_val = None
        if w in ORDINAL_WORDS:
            n_val = ORDINAL_WORDS[w]
        else:
            # Only match digits WITH an ordinal suffix (5th, 12th, 100th)
            # Bare numbers like 10000, 50000 are filter values, not positions
            num_match = re.match(r'^(\d+)(st|nd|rd|th)$', w)
            if num_match:
                n_val = int(num_match.group(1))

        if n_val is not None:
            requested_n = n_val
            # Look at nearby words (within next 2 tokens) for a document type
            for offset in range(1, 3):
                if idx + offset < len(q_words) and q_words[idx + offset] in DOC_TYPE_WORDS:
                    doc_type_found = DOC_TYPE_WORDS[q_words[idx + offset]]
                    break
            break  # use the first number found

    has_position_intent = requested_n is not None and any(
        k in question.lower() for k in [
            "show", "give me", "what is", "display", "get", "find"
        ]
    )

    if has_position_intent:
        doc_type = doc_type_found or active_state.get("active_dataset")
        if doc_type not in ("invoice", "contract", "receipt"):
            doc_type = "invoice"  # safe universal default

        count_sql = f"""
            SELECT COUNT(*) as total
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE d.status = 'done'
            AND r.document_type = '{doc_type}'
        """
        try:
            count_result = await execute_query(count_sql, db)
            total_count = count_result[0].get("total", 0) if count_result else 0

            if requested_n > total_count or requested_n < 1:
                answer = (
                    f"There {'is' if total_count == 1 else 'are'} only {total_count} "
                    f"{doc_type}{'s' if total_count != 1 else ''} — no {requested_n}"
                    f"{_ordinal_suffix(requested_n)} {doc_type} exists."
                )
                save_turn(session_id, "assistant", answer, {"intent": "nth_item_out_of_range"})
                return {
                    "answer": answer,
                    "intent": "nth_item_out_of_range",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": None,
                    "results": [],
                    "count": 0
                }

            offset = requested_n - 1
            nth_sql = f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (d.filename) d.filename,
                        r.extracted_data->>'vendor_name' as vendor,
                        r.extracted_data->>'parties' as parties,
                        r.extracted_data->>'total_amount' as amount,
                        r.extracted_data->>'value' as value,
                        r.extracted_data->>'currency' as currency,
                        r.extracted_data->>'due_date' as due_date,
                        r.extracted_data->>'end_date' as end_date,
                        r.extracted_data->>'issue_date' as issue_date
                    FROM documents d
                    JOIN extraction_results r ON r.doc_id = d.id
                    WHERE r.document_type = '{doc_type}'
                    AND d.status = 'done'
                    ORDER BY d.filename, d.created_at DESC
                ) _sub
                ORDER BY filename DESC NULLS LAST
                OFFSET {offset} LIMIT 1
            """
            is_valid, _ = validate_sql(nth_sql)
            if is_valid:
                nth_results = await execute_query(nth_sql, db)
                answer = _build_list_response(nth_results, question) if nth_results else f"No {doc_type} found at position {requested_n}."
                set_last_results(session_id, nth_results)
                save_turn(session_id, "assistant", answer, {
                    "intent": "nth_item", "count": len(nth_results)
                })
                return {
                    "answer": answer,
                    "intent": "nth_item",
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": nth_sql,
                    "results": nth_results,
                    "count": len(nth_results)
                }
        except Exception as e:
            logger.error(f"[{session_id}] Nth item lookup failed: {e}", exc_info=True)

    # ── Step 5.6: Query operation planner ───
    skip_transform = any(k in question.lower() for k in [
        "above ", "below ", "more than", "less than",
        "this month", "last month", "this year",
        "show all invoices", "list all", "show me all",
        "which vendor", "top vendor", "most paid", "highest total",
        "how much total", "total paid", "sum of", "average",
        "how many", "count of", "breakdown", "percentage",
        "show invoices", "show contracts", "show receipts",
        "find invoices", "find contracts", "find vendors",
        # ── ADD THESE ──
        "show all receipts", "list receipts", "all receipts",
        "eur invoices", "usd invoices", "gbp invoices",
        "invoices only", "only eur", "only usd",
        "overdue", "expiring", "expired", "past due",
        "brightpath", "nordic", "eurodata", "quantumcore",
        "finedge", "bluewave", "techno", "cloudpeak",
        "invoices from", "contracts from", "receipts from",
        "documents from", "show all documents",
        "smallest", "largest", "cheapest", "most expensive",
        "newest", "oldest", "latest", "earliest",
        "failed", "error", "processing",
        "what document", "document types", "how many documents",
        "document type", "what type", "which types",
        "total amount", "total of all", "sum of all",
        "total eur", "total usd", "total spending",
        "grand total", "overall total", "unusual", "suspicious", "anomal", "outlier", "unusual amount",
        "are there any", "fraud", "weird", "abnormal", "missing due", "missing value", "missing vendor", "missing field",
        "no due date", "without due date", "no vendor", "missing date",  "does ", "appear in", "appear in contracts", "appear in invoices",
        "appear in receipts", "is there ", "can i find",
        "exist in", "exists in", "found in", "compare top", "compare bottom", "compare first", "compare last",
        "and show their", "and their contract", "and their invoice",
        "and their receipt", "with their contract", "with their invoice",
        "along with their", "plus their","th invoice", "rd invoice", "nd invoice", "st invoice",
        "th contract", "rd contract", "nd contract", "st contract",
        "th receipt", "rd receipt", "nd receipt", "st receipt",
        "th document", "rd document", "nd document", "st document",
    ])

    # ── Guard: bare/contentless commands are ALWAYS standalone ────────────────
    # Prevents the LLM operation planner from hallucinating a transform
    # intent out of leftover conversation history when the question itself
    # carries no real signal (e.g. "Show", "List", "Get", "Display" alone).
    bare_command_words = {"show", "list", "display", "get", "find", "fetch"}
    question_tokens = [t for t in re.findall(r"[a-z]+", question.lower())]
    is_bare_command = (
        len(question_tokens) <= 1 and
        (not question_tokens or question_tokens[0] in bare_command_words)
    )
    if is_bare_command:
        logger.info(f"[{session_id}] Bare command '{question}' detected — forcing standalone")
        skip_transform = True

    if not skip_transform and not document_type_mismatch and active_state.get("last_sql"):
        operation = plan_query_operation(
            question, active_state, resolved_context, history
        )

    else:
        operation = {
            "type": "standalone",
            "operation": None,
            "params": {},
            "reuse_state": False
        }

    logger.info(f"[{session_id}] Operation plan: {operation}")

    if (
        operation["type"] == "transform"
        and operation.get("operation") not in ("aggregate", None)
        and active_state.get("last_sql")
    ):
        logger.info(f"[{session_id}] Applying transformation to active dataset")
        try:
            transformed_sql = apply_operation_to_sql(
                active_state["last_sql"],
                operation,
                active_state
            )
            is_valid, reason = validate_sql(transformed_sql)
            if is_valid:
                results = await execute_query(transformed_sql, db)
                logger.info(f"[{session_id}] Transform returned {len(results)} rows")

                answer = synthesize_response(
                    original_question=question,
                    rewritten_question=question,
                    results=results,
                    history=history,
                    intent=intent
                )

                update_active_state(session_id, {
                    "sort": operation.get("params", {}).get("sort", {}),
                    "result_count": len(results),
                    "last_sql": transformed_sql
                })
                set_last_results(session_id, results)

                new_focus = build_conversation_focus(
                    intent, question, results, resolved_context, conversation_focus
                )
                set_conversation_focus(session_id, new_focus)

                save_turn(session_id, "assistant", answer, {
                    "intent": intent,
                    "sql": transformed_sql,
                    "original_question": question,
                    "count": len(results),
                    "results_sample": results[:3]
                })

                return {
                    "answer": answer,
                    "intent": intent,
                    "session_id": session_id,
                    "rewritten": None,
                    "sql": transformed_sql,
                    "results": results,
                    "count": len(results)
                }
        except Exception as e:
            logger.error(f"[{session_id}] Transform failed: {e} — falling back to SQL generation")
            try:
                await db.rollback()
                logger.info(f"[{session_id}] Transaction rolled back after transform failure")
            except Exception as rb_err:
                logger.error(f"[{session_id}] Rollback failed: {rb_err}")

    # ── Step 6: Rewrite query ─────────────────────
    rewritten = rewrite_query(question, history, last_metadata, resolved_context)
    logger.info(f"[{session_id}] Rewritten: {rewritten}")

    # ── Step 6.5: Resolve vendor mentions to canonical DB names ──
    try:
        all_vendors = await get_all_vendor_names(db)
        sql_input_question = resolve_vendor_names_in_question(rewritten, all_vendors)
        if sql_input_question != rewritten:
            logger.info(f"[{session_id}] Vendor resolved for SQL: '{rewritten}' -> '{sql_input_question}'")
    except Exception as e:
        logger.error(f"[{session_id}] Vendor resolution failed: {e}")
        sql_input_question = rewritten


    # ── Step 7: Generate SQL ──────────────────────
    sql = generate_sql(sql_input_question, history, resolved_context, all_vendors=all_vendors)
    logger.info(f"[{session_id}] Generated SQL: '{sql}'")

    if not sql:
        answer = synthesize_error(question, "Could not generate SQL")
        save_turn(session_id, "assistant", answer, {"intent": intent})
        return {
            "answer": answer,
            "intent": intent,
            "session_id": session_id,
            "rewritten": rewritten,
            "sql": None,
            "results": [],
            "count": 0
        }

    # ── Step 8: Validate SQL ──────────────────────
    is_valid, reason = validate_sql(sql)
    if not is_valid:
        logger.warning(f"[{session_id}] Invalid SQL: {reason}")
        answer = f"I cannot execute that query for security reasons: {reason}"
        save_turn(session_id, "assistant", answer, {"intent": intent})
        return {
            "answer": answer,
            "intent": intent,
            "session_id": session_id,
            "rewritten": rewritten,
            "sql": None,
            "results": [],
            "count": 0
        }

        # ── Step 9: Execute SQL ───────────────────────
    try:
        results = await execute_query(sql, db)
        logger.info(f"[{session_id}] Query returned {len(results)} rows")

        # ── Deduplicate by filename + document_type ──
        seen = set()
        deduped = []
        for row in results:
            # Create unique key from filename + any ID field available
            key = (
                row.get("filename", "") or
                row.get("invoice_number", "") or
                row.get("doc_id", "")
            )
            if key and key not in seen:
                seen.add(key)
                deduped.append(row)
            elif not key:
                deduped.append(row)  # keep rows without filename (aggregations)
        if len(deduped) < len(results):
            logger.info(f"[{session_id}] After dedup: {len(deduped)} rows")
            results = deduped

    except Exception as e:
        logger.error(f"[{session_id}] Query failed: {e}", exc_info=True)
        answer = synthesize_error(question, str(e))
        save_turn(session_id, "assistant", answer, {
            "intent": intent,
            "sql": sql,
            "error": str(e)
        })
        return {
            "answer": answer,
            "intent": intent,
            "session_id": session_id,
            "rewritten": rewritten,
            "sql": sql,
            "results": [],
            "count": 0,
            "error": str(e)[:200]
        }

    # ── Step 10: Anomaly detection ────────────────
    anomaly_keywords = ["suspicious", "anomal", "outlier", "unusual", "fraud", "weird", "abnormal", "strange amount"]
    is_anomaly_query = any(kw in question.lower() for kw in anomaly_keywords)
    if is_anomaly_query:
        outliers = detect_outliers(results)
        duplicates = detect_duplicates(results)
        if outliers or duplicates:
            results = outliers + duplicates
            logger.info(f"[{session_id}] Anomalies found: {len(results)}")

    # Deduplicate results
    results = deduplicate_results(results)

    # Hard cap
    # results = results[:50]
    logger.info(f"[{session_id}] After dedup: {len(results)} rows")

    # ── Step 11: Synthesize response ──────────────
    answer = synthesize_response(
        original_question=question,
        rewritten_question=rewritten,
        results=results,
        history=history,
        intent=intent
    )

   # ── Step 12: Save to memory + update state ────
    new_focus = build_conversation_focus(
        intent, question, results, resolved_context, conversation_focus
    )

    # Force duplicate focus when applicable
    duplicate_keywords = ["duplicate", "repeat", "same invoice", "appears again"]
    if any(kw in question.lower() for kw in duplicate_keywords):
        new_focus["topic"] = "duplicate_invoices"
        new_focus["query_type"] = "duplicate_detection"

    set_conversation_focus(session_id, new_focus)
    set_last_results(session_id, results)

    # ── Detect active dataset from SQL — universal, not from context ──
    # This is the key fix — detect dataset from actual SQL generated
    sql_lower = sql.lower()
    if "document_type = 'invoice'" in sql_lower:
        detected_dataset = "invoice"
    elif "document_type = 'contract'" in sql_lower:
        detected_dataset = "contract"
    elif "document_type = 'receipt'" in sql_lower:
        detected_dataset = "receipt"
    elif "document_type = 'report'" in sql_lower:
        detected_dataset = "report"
    else:
        detected_dataset = resolved_context.get("document_type") or \
                          new_focus.get("doc_type") or \
                          active_state.get("active_dataset", "invoice")

    # ── Update active resultset state ──
    new_state = {
        "intent": intent,
        "active_dataset": detected_dataset,
        "filters": {
            k: v for k, v in {
                "currency": resolved_context.get("currency"),
                "vendor": resolved_context.get("vendor_name"),
                "time_period": resolved_context.get("time_period"),
                "amount_gt": resolved_context.get("amount_reference")
            }.items() if v
        },
        "entities": {
            k: v for k, v in {
                "vendor": new_focus.get("vendor"),
                "currency": new_focus.get("currency"),
                "amount": new_focus.get("amount"),
                "filename": new_focus.get("filename")
            }.items() if v
        },
        "result_count": len(results),
        "shown_count": min(len(results), 10),
        "last_sql": sql,
        "query_scope": "filtered" if resolved_context.get("vendor_name") or
                       resolved_context.get("currency") else "all"
    }
    set_active_state(session_id, new_state)
    logger.info(f"[{session_id}] Active state saved: dataset={detected_dataset} sql_len={len(sql)}")

    save_turn(session_id, "assistant", answer, {
        "intent": intent,
        "sql": sql,
        "original_question": question,
        "rewritten_question": rewritten,
        "count": len(results),
        "results_sample": results[:3]
    })

    return {
        "answer": answer,
        "intent": intent,
        "session_id": session_id,
        "rewritten": rewritten if rewritten != question else None,
        "sql": sql,
        "results": results[:20],
        "count": len(results)
    }