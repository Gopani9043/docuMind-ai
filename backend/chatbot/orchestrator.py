
import logging
import os

import os
print("ORCHESTRATOR FILE:", os.path.abspath(__file__))
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

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

            fuzzy = find_matches(candidate, all_vendors, threshold=0.55)
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
    complexity_indicators = [
        "and then", "also show", "summarize", "trend",
        "as well as", "additionally", "furthermore",
        "compare and", "show me both"
    ]
    has_indicator = any(ind in q for ind in complexity_indicators)
    has_many_commas = question.count(",") > 2
    return has_indicator or has_many_commas


# ── Helper: Question Decomposer ───────────────────────────────────────────────

def decompose_question(question: str) -> list:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template("""
Break this complex question into 2-4 simple atomic sub-questions.
Each sub-question must be answerable with a single SQL query.
Return ONLY a numbered list. No explanation.

Question: {question}

Sub-questions:
""")
    try:
        chain = prompt | llm
        result = chain.invoke({"question": question})
        lines = result.content.strip().split("\n")
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

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )

    try:
        chain = OPERATION_PLANNER_PROMPT | llm
        response = chain.invoke({
            "question": question,
            "active_state": json.dumps(active_state, default=str),
            "history": history or "No history"
        })

        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        operation = json.loads(content.strip())
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
        "duplicate invoice"
    ]
    if any(kw in question.lower() for kw in repeat_keywords):
        logger.info(f"[{session_id}] Repeat invoice detection triggered")
        repeat_sql = """
            SELECT
                r.extracted_data->>'invoice_number' as invoice_number,
                r.extracted_data->>'vendor_name' as vendor,
                r.extracted_data->>'total_amount' as amount,
                r.extracted_data->>'currency' as currency,
                COUNT(*) as repeat_count
            FROM documents d
            JOIN extraction_results r ON r.doc_id = d.id
            WHERE d.status = 'done'
            AND r.document_type = 'invoice'
            AND r.extracted_data->>'invoice_number' IS NOT NULL
            AND r.extracted_data->>'invoice_number' != ''
            GROUP BY
                r.extracted_data->>'invoice_number',
                r.extracted_data->>'vendor_name',
                r.extracted_data->>'total_amount',
                r.extracted_data->>'currency'
            HAVING COUNT(*) > 1
            ORDER BY repeat_count DESC
            LIMIT 20
        """
        is_valid, _ = validate_sql(repeat_sql)
        if is_valid:
            try:
                repeat_results = await execute_query(repeat_sql, db)
                if repeat_results:
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

                # Update active state
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

    if any(kw in question.lower() for kw in overdue_keywords):
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

    if any(kw in question.lower() for kw in expiring_keywords):
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
        "sum of all invoices", "total amount all invoices"
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

    # ── Step 4: Decompose if complex ─────────────
    if is_complex_question(question):
        logger.info(f"[{session_id}] Complex question — decomposing")
        sub_questions = decompose_question(question)
        sub_results = []

        for sub_q in sub_questions:
            try:
                sub_context = resolve_context(
                    sub_q, history, last_metadata,
                    last_results=last_results,
                    conversation_focus=conversation_focus
                )
                sub_rewritten = rewrite_query(sub_q, history, last_metadata, sub_context)
                sub_sql = generate_sql(sub_rewritten, history, sub_context, all_vendors=all_vendors)

                if not sub_sql:
                    continue

                is_valid, reason = validate_sql(sub_sql)
                if not is_valid:
                    continue

                sub_data = await execute_query(sub_sql, db)
                sub_results.append({
                    "question": sub_q,
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
        if "second" in q_lower:   index = 1
        elif "third" in q_lower:  index = 2
        elif "fourth" in q_lower: index = 3
        elif "fifth" in q_lower:  index = 4
        elif "last" in q_lower:   index = -1

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
        "grand total", "overall total",
    ])

    if not skip_transform and active_state.get("last_sql"):
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
    anomaly_keywords = ["suspicious", "anomal", "duplicate", "outlier", "unusual", "fraud"]
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