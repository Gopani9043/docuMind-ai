import logging
from langchain_core.prompts import ChatPromptTemplate
from chatbot.llm_provider import invoke_with_fallback

logger = logging.getLogger(__name__)

SPECIAL_INTENT_PROMPT = ChatPromptTemplate.from_template("""
You are an intent router for a document analytics chatbot.
Classify the user's question into EXACTLY ONE category based on MEANING,
not exact keywords. Users phrase the same intent in many different ways —
map the underlying meaning regardless of wording.

CATEGORIES:

biggest_bills
  User wants their largest invoices/bills/payments, ranked by TRUE
  financial size (must work correctly across multiple currencies).
  Examples: "What are my biggest bills?", "Show my priciest invoices",
  "Which invoice cost the most?", "Top expenses", "What did I pay most for?"

  IMPORTANT EXCLUSION: "Show invoices above/below/over/under X EUR/USD"
  is a THRESHOLD FILTER, not a request to rank by size — classify as
  "none" even though it mentions an amount and the word "invoice".
  The user wants ALL invoices matching a condition, not a ranked subset.
  
spending_trend
  User wants to know if total spending is going up, down, or stable
  over time — a DIRECTION, not a list.
  Examples: "Is our spending increasing or decreasing?", "Are we spending
  more than before?", "Is spending trending up?", "Are costs going up?",
  "How has spending changed over time?",  "Which month
  had the highest invoice total?", "What was our best/worst month for
  spending?", "Which month did we spend the most/least?", "Top spending month"

something_interesting
  User wants an open-ended summary of notable facts — not a specific lookup.
  Examples: "Show me something interesting", "Surprise me", "Any insights?",
  "What stands out?", "Give me a quick overview", "Tell me something I should know"

missing_data
  User wants to know about incomplete/missing fields (data quality check).
  Examples: "Any missing data?", "What's incomplete?", "Data quality check",
  "Are there gaps in my data?", "Is anything incomplete?"

vendor_risk
  User wants to know which vendors are concerning/risky/problematic.
  Examples: "Which vendors should I worry about?", "Any risky vendors?",
  "Should I be concerned about anything?", "Any red flags?", "Any vendors to watch?"

most_important_vendor
  User wants a vendor positioned at ANY point in an importance ranking
  that combines spend, frequency, and relationship strength — the single
  top vendor, the single least important vendor, a middle/average/medium
  vendor, an Nth-ranked vendor, a tier (high/medium/low), or a comparison
  between two vendors' importance. Route ANY question about vendor
  importance or criticality here, regardless of which position is asked for.
  Examples: "Who is our most important vendor?", "Which vendor matters most?",
  "Who is our key supplier?", "Which vendor is most critical to us?",
  "Who is our least important vendor?", "Which vendor matters least?",
  "Who is our medium importance vendor?", "Which vendor is moderately
  important?", "Who is our second most important vendor?", "Show our
  top 3 most important vendors", "Is vendor X more important than vendor Y?"

DISAMBIGUATION RULE — CRITICAL:
If the question starts with "which month", "what month", "which period",
"which quarter" — the user is ranking MONTHS/PERIODS, not individual
invoices or vendors. Classify as spending_trend, even if the question
also mentions "invoice", "bill", "total", or "amount" later in the
sentence. The grammatical subject ("month") determines the category,
not other nouns that appear afterward.

Examples:
- "Which month had the second highest invoice?" → spending_trend
  (asking to rank MONTHS by total, not asking which single invoice)
- "Which month had the highest invoice total?" → spending_trend
- "What was our biggest invoice?" → biggest_bills
  (no "which month" prefix — this genuinely asks about one invoice)

none
  The question doesn't match any category above — simple lookups, filters,
  lists, comparisons, single facts, etc.
  Examples: "Show all invoices", "Total spending in EUR", "Which vendor
  paid the most", "Compare this month vs last month"

CONVERSATION HISTORY:
{history}

QUESTION:
{question}

Return ONLY one word: biggest_bills, spending_trend, something_interesting, missing_data, vendor_risk, most_important_vendor, or none
No explanation. No punctuation. No markdown.
""")


def classify_special_intent(question: str, history: str = "") -> str:
    """
    Semantic router for special analytical intents.
    Replaces fragile keyword-list matching with true meaning-based
    classification — works regardless of how the user phrases the question.
    """
    try:
        result = invoke_with_fallback(
            lambda llm: SPECIAL_INTENT_PROMPT | llm,
            {"question": question, "history": history or "No history"}
        )
        category = result.strip().lower().strip(".")
        valid = {
            "biggest_bills", "spending_trend", "something_interesting",
            "missing_data", "vendor_risk", "most_important_vendor", "none"
        }
        if category not in valid:
            logger.warning(f"Special intent classifier returned unexpected value: '{category}' — defaulting to none")
            return "none"
        logger.info(f"Special intent classified: '{question}' -> {category}")
        return category
    except Exception as e:
        logger.error(f"Special intent classification failed: {e}")
        return "none"