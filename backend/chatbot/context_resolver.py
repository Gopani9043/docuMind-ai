import os
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from services.vendor_matcher import normalize

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
)

CONTEXT_PROMPT = ChatPromptTemplate.from_template("""
You are a context resolver for a document analytics chatbot.
Extract key context from the conversation to help answer the current question.

CONVERSATION HISTORY:
{history}

LAST QUERY RESULTS SUMMARY:
{last_results}

CURRENT QUESTION:
{question}

CRITICAL RESOLUTION RULES — always apply these before extracting context:

1. If the previous query was an AGGREGATION (total, sum, count, average, top vendor, highest spending)
   AND the current question is "which invoice is this?" or "show me the invoice" or "which one is this?"
   or "what is it?" or "show me that" or "which one?" →
   ALWAYS resolve to: the largest single record from the previously mentioned vendor/type.
   Set intent_override to "show_largest_single_record".
   NEVER interpret this as closest-to-average.

2. "which one is this" / "which one" / "that one" after an aggregation result →
   means the top individual record by amount DESC, not closest to average.

3. "which invoice is this" after a vendor total query →
   means the largest single invoice from that vendor ORDER BY amount DESC LIMIT 1.

4. "show me the invoice" / "show me that invoice" / "what invoice" →
   largest invoice from the vendor in context, not a search or average match.

5. Only use closest-to-average logic when user explicitly says
   "closest to", "nearest to", "around", "approximately".

Extract and return a JSON object with these fields:
{{
  "document_type": "invoice|contract|receipt|report|null",
  "vendor_name": "vendor name if referenced or null",
  "currency": "currency if referenced or null",
  "amount_reference": "specific amount referenced like 43482 or null",
  "time_period": "this month|last month|this week|null",
  "comparison_requested": true or false,
  "previous_query_type": "aggregation|single_record|list|null",
  "intent_override": "show_largest_single_record|null",
  "resolved_references": "what vague terms like this/that/those refer to — be specific, name the actual vendor/document/amount"
}}

Return ONLY valid JSON. No explanation.
""")

def resolve_context(question: str, history: str, last_metadata: dict) -> dict:
    """
    Extract context clues from conversation history.
    Returns structured context for query generation.
    """
    # Default context
    default_context = {
        "document_type": None,
        "vendor_name": None,
        "currency": None,
        "amount_reference": None,
        "time_period": None,
        "comparison_requested": False,
        "previous_query_type": None,
        "resolved_references": None
    }

    if not history:
        return default_context

    # Fast rule-based resolution
    q_lower = question.lower()

    # Check for comparison
    if any(w in q_lower for w in ["compare", "vs", "versus", "last month", "previous"]):
        default_context["comparison_requested"] = True

    # Check for time references
    if "this month" in q_lower:
        default_context["time_period"] = "this month"
    elif "last month" in q_lower:
        default_context["time_period"] = "last month"
    elif "this week" in q_lower:
        default_context["time_period"] = "this week"

    # Extract from last metadata
    if last_metadata:
        results = last_metadata.get("results_sample", [])
        if results:
            first = results[0] if results else {}
            if "vendor" in first and first.get("vendor"):
                raw_vendor = first.get("vendor")
                default_context["vendor_name"] = normalize(raw_vendor)  # ← uses fuzzy matcher
            if "currency" in first:
                default_context["currency"] = first.get("currency")

    # Use LLM for complex references
    vague_terms = ["this", "that", "those", "it", "the highest", "the lowest",
                   "same", "similar", "previous", "above", "that average",
                   "closest", "nearest", "compared"]

    if any(term in q_lower for term in vague_terms):
        try:
            import json
            chain = CONTEXT_PROMPT | llm
            result = chain.invoke({
                "question": question,
                "history": history,
                "last_results": str(last_metadata)[:500]
            })
            content = result.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            resolved = json.loads(content.strip())
            default_context.update(resolved)
            logger.info(f"Resolved context: {resolved}")
        except Exception as e:
            logger.error(f"Context resolution failed: {e}")

    return default_context