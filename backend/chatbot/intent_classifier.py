import logging
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from chatbot.llm_provider import invoke_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)

INTENT_PROMPT = ChatPromptTemplate.from_template("""
Classify this question into exactly ONE intent.

INTENTS:
- sql_analytics: needs database query (list, filter, count, sum, compare, rank, aggregate, trend, risk)
- sql_search: search documents by content or text
- document_qa: question about a specific document's content
- general: general question about OCR, AI, document processing
- greeting: hello, hi, thanks, help, what can you do
- clarification: asking to explain or clarify previous answer
- reset: clear chat, start over, forget everything

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {question}

Rules:
- If question references previous results ("the highest one", "those invoices", "that vendor") → sql_analytics
- If question asks about document contents → document_qa
- If question is about system capabilities → general
- If question compares two vendors, amounts, or periods → sql_analytics
- If question asks about importance, risk, concern, worry, or value of vendors/invoices → sql_analytics
- If question asks "is X more/less/better/bigger than Y" → sql_analytics
- If question asks "should I be worried/concerned" → sql_analytics
- If question asks "who matters more", "which is better", "how do they compare" → sql_analytics
- NEVER classify vendor comparison or spending concern questions as greeting

Examples that are sql_analytics:
- "Is BrightPath more important than QuantumCore?"
- "Should I be worried about our spending?"
- "Which vendor matters more?"
- "How does BrightPath compare to FinEdge?"
- "Is our contract spend increasing?"
- "Who is riskier, EuroData or QuantumCore?"
- "Do they just receive more invoices or are they actually more important?"

Return ONLY the intent name. Nothing else.
""")

VALID_INTENTS = {
    "sql_analytics", "sql_search", "document_qa",
    "general", "greeting", "clarification", "reset"
}


def classify_intent(question: str, history: str = "") -> str:
    """Classify the intent of a user question."""

    q_lower = question.lower().strip()

    # ── Fast path: greetings ──────────────────────────────────────────────────
    greeting_triggers = ["hello", "hi ", "hey ", "thanks", "thank you", "bye", "what can you"]
    # Only trigger greeting if the ENTIRE question is greeting-like
    # (short, no analytical content) — prevents "hi, which vendor..." → greeting
    if any(w in q_lower for w in greeting_triggers) and len(q_lower.split()) <= 5:
        return "greeting"

    # ── Fast path: reset ──────────────────────────────────────────────────────
    if any(w in q_lower for w in ["clear chat", "reset", "start over", "forget everything", "new conversation"]):
        return "reset"

    # ── Fast path: comparison/concern questions — always sql_analytics ────────
    # These are natural language questions that lack document keywords but are
    # clearly analytical — the LLM classifier sometimes misroutes these.
    analytical_patterns = [
        "more important", "less important", "most important",
        "more than", "less than", "better than", "worse than",
        "should i be worried", "should we be worried", "worried about",
        "concerned about", "is it too", "too much", "too high",
        "how do they compare", "how does", "which is better",
        "which matters", "who matters", "who is more", "who is less",
        "is spending", "are we spending", "do they just", "or do they",
        "more invoices", "more contracts", "more receipts",
        "important than", "riskier than", "bigger than", "smaller than",
    ]
    if any(p in q_lower for p in analytical_patterns):
        return "sql_analytics"

    # ── Fast path: explicit SQL keywords ─────────────────────────────────────
    sql_keywords = [
        "show", "list", "find", "get", "how many", "total", "sum",
        "average", "top", "highest", "lowest", "compare", "filter",
        "invoice", "contract", "receipt", "vendor", "amount", "currency",
        "overdue", "expiring", "missing", "uploaded", "this month",
        "last month", "this week", "trend", "breakdown", "distribution",
        "spelling", "duplicate", "similar", "fuzzy", "suspicious", "anomaly",
        "document type", "what type", "which type", "how many document",
        "what document", "status", "failed", "processing", "done",
        "how much", "who", "which", "what is the", "what are",
        "when was", "when were", "largest", "smallest", "newest", "oldest",
        "latest", "earliest", "repeat", "appears", "paid", "spent",
        "document", "report", "file", "pdf", "important", "risk",
        "worry", "concern", "spending", "biggest", "most",
    ]
    if any(kw in q_lower for kw in sql_keywords):
        return "sql_analytics"

    # ── LLM fallback for genuinely ambiguous cases ────────────────────────────
    try:
        intent = invoke_with_fallback(
            lambda llm: INTENT_PROMPT | llm,
            {
                "question": question,
                "history": history or "No history"
            }
        )
        intent = intent.strip().lower()
        return intent if intent in VALID_INTENTS else "sql_analytics"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "sql_analytics"