import json
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
  User wants to see their largest individual invoice DOCUMENTS ranked by
  financial size — the output is a list of specific files/invoices.
  The question is about single transactions, not vendor totals.

  Examples:
  "What are my biggest bills?"
  "Show my most expensive invoices"
  "Which invoice cost the most?"
  "Top 5 largest invoices"
  "Show the priciest documents"
  "What is the single largest payment I made?"

  NOT this category — these ask for vendor TOTALS across all invoices,
  not individual documents (route these to vendor_spend_ranking instead):
  "Which vendor did I pay the most?"
  "Top vendor by total spending"
  "Which company costs us the most overall?"
  "Who is our biggest supplier by spend?"
  "Welcher Lieferant hat am meisten bekommen?"

biggest_contracts
  User wants their largest INDIVIDUAL contracts ranked by value.
  Examples: "What's my biggest contract?", "Which contract has the highest value?",
  "Show my largest agreements", "Most expensive contract", "Top contracts by value"

spending_trend
  User wants to know if total spending is going up, down, or stable over time.
  Can be for invoices OR contracts — detect which from the question.
  over time — a DIRECTION, not a list.
  Examples: "Is our spending increasing or decreasing?", "Are we spending
  more than before?", "Is spending trending up?", "Are costs going up?",
  "How has spending changed over time?",  "Which month
  had the highest invoice total?", "What was our best/worst month for
  spending?", "Which month did we spend the most/least?", "Top spending month", "How has contract value changed over time?",
  "Show contract spending trend over time"

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
  This is a BLENDED score (spend + frequency + contract presence) — use
  this ONLY when the question is specifically about "importance" or
  "criticality", not when it's purely about money paid.
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

vendor_spend_ranking
  User wants vendors (or contract parties/companies) ranked PURELY by
  how much money was paid/spent/billed — no composite scoring, just raw
  financial totals. Must work correctly across multiple currencies.
  This is the category for ALL vendor-total and vendor-aggregate
  questions, whether about invoices or contracts.

  Examples (invoices): "Which vendor did I pay the most?", "Top 5 vendors
  by total paid", "Who do we spend the most with?", "Show top 3 vendors
  in EUR", "Which vendor did we pay the least?", "Second highest
  spending vendor", "Who is our biggest supplier by spend?",
  "Compare BrightPath vs FinEdge total spending", "Compare X and Y
  spending", "Is BrightPath bigger than FinEdge in terms of spend?"

  Examples (contracts): "Show vendor spend ranking for contracts",
  "Which company has the highest contract value?", "Rank companies by
  contract value", "Show contract totals by party"

  Examples (statistical positions, any scope): "Which vendor's spending
  is closest to average?", "Show the vendor with median spending",
  "Which vendor spends an average amount?"

  IMPORTANT: comparing exactly 2 named vendors' SPEND is still pure
  financial ranking — classify here, not as most_important_vendor.

  DO NOT confuse with most_important_vendor — that blends spend with
  frequency and contract presence. This category is PURE spend/total
  value, nothing else, even if the word "important" never appears.

none
  The question doesn't match any category above — simple lookups,
  filters, lists, comparisons of non-vendor data, single facts, etc.
  Examples:
  "Show all invoices"
  "Total spending in EUR"
  "Compare this month vs last month"
  "Show contracts expiring soon"

Also detect if the question carries a CONCERN or WORRY tone —
the user is anxious about the data, not just curious.

Examples of concern tone (concern_tone=true):
- "Should I be worried about how much we're spending?"
- "Is our spending getting out of control?"
- "Are costs too high?"
- "Is this sustainable?"
- "Should I be alarmed?"
- "Is something wrong with our spending?"

Examples of neutral tone (concern_tone=false):
- "Show spending trend"
- "Is spending increasing or decreasing?"
- "Show me invoice trend over time"

CONVERSATION HISTORY:
{history}

QUESTION:
{question}

Return ONLY valid JSON with no explanation, no markdown, no backticks:
{{
  "intent": "biggest_bills|biggest_contracts|spending_trend|something_interesting|missing_data|vendor_risk|most_important_vendor|vendor_spend_ranking|none",
  "concern_tone": true or false
}}
""")

VALID_INTENTS = {
    "biggest_bills", "biggest_contracts", "spending_trend",
    "something_interesting", "missing_data", "vendor_risk",
    "most_important_vendor", "vendor_spend_ranking", "none"
}


def classify_special_intent(question: str, history: str = "") -> tuple:
    """
    Returns (intent: str, concern_tone: bool).
    intent: one of VALID_INTENTS
    concern_tone: True if user expressed worry/anxiety, not just curiosity
    """
    try:
        result = invoke_with_fallback(
            lambda llm: SPECIAL_INTENT_PROMPT | llm,
            {"question": question, "history": history or "No history"}
        )
        content = result.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        parsed = json.loads(content)
        intent = parsed.get("intent", "none").strip().lower()
        concern_tone = bool(parsed.get("concern_tone", False))

        if intent not in VALID_INTENTS:
            logger.warning(f"Unexpected intent '{intent}' — defaulting to none")
            intent = "none"

        logger.info(f"Special intent: '{question}' -> {intent} (concern={concern_tone})")
        return intent, concern_tone

    except Exception as e:
        logger.error(f"Special intent classification failed: {e}")
        return "none", False