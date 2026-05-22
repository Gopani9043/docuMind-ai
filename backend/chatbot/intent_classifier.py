import os
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

INTENT_PROMPT = ChatPromptTemplate.from_template("""
Classify this question into exactly ONE intent.

INTENTS:
- sql_analytics: needs database query (list, filter, count, sum, compare, rank, aggregate)
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

Return ONLY the intent name. Nothing else.
""")

VALID_INTENTS = {
    "sql_analytics", "sql_search", "document_qa",
    "general", "greeting", "clarification", "reset"
}


def classify_intent(question: str, history: str = "") -> str:
    """Classify the intent of a user question."""

    # Fast rule-based classification first
    q_lower = question.lower().strip()

    # Greetings
    if any(w in q_lower for w in ["hello", "hi ", "hey", "thanks", "thank you", "bye", "what can you"]):
        return "greeting"

    # Reset
    if any(w in q_lower for w in ["clear chat", "reset", "start over", "forget everything", "new conversation"]):
        return "reset"

    # SQL keywords
    sql_keywords = [
        "show", "list", "find", "get", "how many", "total", "sum",
        "average", "top", "highest", "lowest", "compare", "filter",
        "invoice", "contract", "receipt", "vendor", "amount", "currency",
        "overdue", "expiring", "missing", "uploaded", "this month",
        "last month", "this week", "trend", "breakdown", "distribution"
    ]
    if any(kw in q_lower for kw in sql_keywords):
        return "sql_analytics"

    # Fall back to LLM for ambiguous cases
    try:
        chain = INTENT_PROMPT | llm
        result = chain.invoke({
            "question": question,
            "history": history or "No history"
        })
        intent = result.content.strip().lower()
        return intent if intent in VALID_INTENTS else "sql_analytics"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "sql_analytics"