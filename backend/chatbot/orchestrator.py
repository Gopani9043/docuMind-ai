import logging
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

from chatbot.memory_manager import (
    save_turn, get_recent_context,
    get_last_metadata, clear_session,
    get_session_stats
)
from chatbot.intent_classifier import classify_intent
from chatbot.context_resolver import resolve_context
from chatbot.query_rewriter import rewrite_query
from chatbot.query_generator import generate_sql, validate_sql
from chatbot.response_synthesizer import (
    synthesize_response, synthesize_error, synthesize_general
)
from services.anomaly_detector import detect_outliers, detect_duplicates
from services.vendor_matcher import find_matches, normalize

logger = logging.getLogger(__name__)


# ── Helper: Execute SQL ───────────────────────────────────────────────────────

async def execute_query(sql: str, db: AsyncSession) -> list:
    """Safely execute SQL and return results as list of dicts."""
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


# ── Helper: Complexity Check ──────────────────────────────────────────────────

def is_complex_question(question: str) -> bool:
    """Check if question needs decomposition into sub-questions."""
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
    """Break complex question into atomic sub-questions using LLM."""
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


# ── Main Orchestrator ─────────────────────────────────────────────────────────

async def process_message(
    question: str,
    session_id: str,
    db: AsyncSession
) -> dict:
    """
    Main orchestrator — processes every message through the full pipeline.

    Pipeline:
    1.  Load memory context
    2.  Classify intent
    3.  Handle reset / greeting / general / clarification
    4.  Check complexity — decompose if needed
    5.  Resolve context (vague references)
    6.  Rewrite query
    7.  Generate SQL
    8.  Validate SQL
    9.  Execute SQL
    10. Anomaly detection (if relevant)
    11. Synthesize response
    12. Save to memory
    """
    logger.info(f"[{session_id}] Processing: {question}")

    # ── Step 1: Load memory ───────────────────────
    history = get_recent_context(session_id)
    last_metadata = get_last_metadata(session_id)

    # Save user turn immediately
    save_turn(session_id, "user", question)

    # ── Step 2: Classify intent ───────────────────
    intent = classify_intent(question, history)
    logger.info(f"[{session_id}] Intent: {intent}")

    # ── Step 3a: Handle reset ─────────────────────
    if intent == "reset":
        clear_session(session_id)
        return {
            "answer": "Conversation cleared! Starting fresh. What would you like to know about your documents?",
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
        answer = synthesize_response(
            question, question, [], history, "clarification"
        )
        save_turn(session_id, "assistant", answer, {
            "intent": intent
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

    # ── Step 4: Decompose if complex ─────────────
    if is_complex_question(question):
        logger.info(f"[{session_id}] Complex question — decomposing")
        sub_questions = decompose_question(question)
        sub_results = []

        for sub_q in sub_questions:
            try:
                sub_context = resolve_context(sub_q, history, last_metadata)
                sub_rewritten = rewrite_query(sub_q, history, last_metadata, sub_context)
                sub_sql = generate_sql(sub_rewritten, history, sub_context)

                if not sub_sql:
                    logger.warning(f"[{session_id}] No SQL for sub-question: {sub_q}")
                    continue

                is_valid, reason = validate_sql(sub_sql)
                if not is_valid:
                    logger.warning(f"[{session_id}] Invalid sub-SQL: {reason}")
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
    resolved_context = resolve_context(question, history, last_metadata)
    logger.info(f"[{session_id}] Context: {resolved_context}")

    # ── Step 6: Rewrite query ─────────────────────
    rewritten = rewrite_query(
        question, history, last_metadata, resolved_context
    )
    logger.info(f"[{session_id}] Rewritten: {rewritten}")

    # ── Step 7: Generate SQL ──────────────────────
    sql = generate_sql(rewritten, history, resolved_context)

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
    if any(kw in question.lower() for kw in anomaly_keywords):
        outliers = detect_outliers(results)
        duplicates = detect_duplicates(results)
        if outliers or duplicates:
            results = outliers + duplicates
            logger.info(f"[{session_id}] Anomalies found: {len(results)}")

    # ── Step 11: Synthesize response ──────────────
    answer = synthesize_response(
        original_question=question,
        rewritten_question=rewritten,
        results=results,
        history=history,
        intent=intent
    )

    # ── Step 12: Save to memory ───────────────────
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
        "results": results,
        "count": len(results)
    }