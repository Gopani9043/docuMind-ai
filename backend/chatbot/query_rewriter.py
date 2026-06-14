import os
import json
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    temperature=0,
)

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are a query rewriter for a document analytics chatbot.
Rewrite the vague follow-up question into a clear, standalone question
using the conversation history and context provided.

CONVERSATION HISTORY:
{history}

LAST RESULTS SUMMARY:
{last_results}

RESOLVED CONTEXT:
{resolved_context}

CURRENT QUESTION:
{question}

CRITICAL — CHECK INTENT OVERRIDE FIRST:
Before applying any rewriting rules, check resolved_context for intent_override.

If intent_override is "duplicate_followup":
- Previous query was about duplicate invoices
- "how many times" → "How many times does [invoice] from [vendor] appear?"
- "show them" → "Show all duplicate invoices"
- "which one is newest" → "Which duplicate invoice was uploaded most recently?"
- "give me vendor name" → "What is the vendor name of the most repeated invoice?"

If intent_override is "repetition_count":
- Return exactly: "How many times does the most repeated invoice appear?"

If intent_override is "list_navigation":
- "the second one" → use last_results[1] directly, no SQL needed
- "the third one" → use last_results[2] directly

If intent_override is "show_largest_single_record":
- ONLY apply when conversation_focus is NOT duplicate_invoices
- ONLY apply when NO duplicate context exists in history
- Rewrite to: "Show the single largest invoice from [vendor] ordered by amount descending limit 1"

REWRITING RULES:
- "the highest one" → "show the invoice with highest amount"
- "those invoices" → "show invoices with same filters as previous query"
- "that vendor" → "show documents from [vendor from previous result]"
- "compare with last month" → "compare previous subject this month vs last month"
- "what about contracts" → "show same analysis but for contracts"
- "which ones are overdue" → "show overdue invoices from previous filters"
- "show me more" → "show more results from previous query"
- "that average" → "use the average value from previous answer"
- "the closest one" → "find the invoice closest to amount from previous answer"
- "which invoice is this" + duplicate context → "What is the most repeated invoice?"
- "which invoice is this" + NO duplicate context + has vendor → "Show largest invoice from [vendor] ordered by amount descending limit 1"
- "show me the invoice" + duplicate context → "Show all duplicate invoices"
- "which one" + duplicate context → "Which invoice repeats most?"
- "which one" + NO duplicate context → "Show top record by amount descending limit 1"
- "when was it uploaded" + context has vendor → "Show filename and created_at for [vendor] invoice"
- "show related contracts" + context has vendor → "Show contracts where parties contains [vendor]"
- "give me vendor name" + duplicate context → "What is the vendor of the most repeated invoice?"
- If question is already clear and specific → return it unchanged
FILTER RULES — CRITICAL:
- "only [vendor]" or "only [vendor] from above" → "Show all invoices above 10000 EUR from [vendor]"
  NEVER rewrite to "largest" — user wants ALL filtered results
- "only above X" → add amount filter to previous query
- "only EUR" or "only USD" → add currency filter to previous query
- "smallest one" or "cheapest one" AFTER a filter query → 
  "Show the smallest invoice from [vendor] above [amount] [currency]"
  ALWAYS carry forward ALL previous filters — vendor, currency, amount threshold
- "largest one" AFTER a filter query →
  "Show the largest invoice from [vendor] above [amount] [currency]"

Return ONLY the rewritten question. No explanation. No quotes. No markdown.

Rewritten question:
""")

# Questions that are clearly standalone — skip rewriting
STANDALONE_PATTERNS = [
    "show all", "list all", "how many total",
    "give me all", "find all", "what is the total",
    "top 10", "top 5", "show invoices", "show contracts",
    "show receipts", "show documents",
    "second largest", "second biggest", "second highest", "second most expensive",
    "second smallest", "second cheapest", "second lowest",
    "second newest", "second oldest",
    "third largest", "third highest", "third smallest", "third cheapest",
    "fourth largest", "fifth largest",
    "most expensive", "cheapest invoice", "cheapest contract",
    "newest invoice", "oldest invoice", "newest contract", "oldest contract",
    "show the 10", "show the 5", "show the 3",
    # Comparison queries — always fresh SQL, never rewrite
    "compare eur", "compare usd", "compare gbp",
    "compare brightpath", "compare finedge", "compare nordic",
    "eur vs usd", "usd vs eur", "vs usd", "vs eur",
    "compare average", "compare total",

]

# Vague reference indicators — always trigger rewrite
VAGUE_INDICATORS = [
    "the highest", "the lowest", "that one", "those", "this one",
    "that invoice", "that vendor", "that contract", "the same",
    "compare", "what about", "which ones", "show me more",
    "that average", "closest", "nearest", "the best", "the worst",
    "previous", "above mentioned", "it ", "them ", "they ",
    "which invoice is this", "which one is this", "what is this",
    "show me the invoice", "show me that", "which one",
    "when was it", "when was it uploaded", "show related",
    "is this", "what invoice", "what contract",
    "how many times", "repeat", "how often", "appear",
    "next one", "previous one",         # removed "the second", "the third"
    "show them", "list them", "what currency", "how much",
    "give me", "vendor name", "this invoice", "this vendor",
    "only ", "just ", "from them", "show all from",
    "their invoices", "their contracts", "all from",
    "from that vendor", "from same vendor","only ",
    # removed "newest", "oldest", "latest", "earliest"
]



def rewrite_query(
    question: str,
    history: str,
    last_metadata: dict,
    resolved_context: dict
) -> str:
    """
    Rewrite vague follow-up questions into clear standalone questions.
    Returns original question if already clear.
    """
    q_lower = question.lower().strip()
    ordinal_superlative = [
        "second largest", "second biggest", "second highest", "second most",
        "second smallest", "second cheapest", "second lowest",
        "second newest", "second oldest",
        "third largest", "third highest", "third smallest",
        "what is the second", "what is the third",
    ]
    # Never rewrite comparison queries — they need fresh SQL
    if " vs " in q_lower or "compare " in q_lower:
        logger.info(f"Comparison query — no rewrite: '{question}'")
        return question

    if any(p in q_lower for p in ordinal_superlative):
        logger.info(f"Ordinal+superlative — no rewrite: '{question}'")
        return question
    intent_override = resolved_context.get("intent_override") if resolved_context else None
    conversation_focus = resolved_context.get("conversation_focus", "") or ""
    
    # ── Check duplicate focus BEFORE any override ──
    is_duplicate_focus = any(k in str(conversation_focus).lower() for k in
                             ["duplicate", "repeat", "duplicate_invoices"])

    # ── show_largest_single_record ONLY when NOT in duplicate context ──
    if intent_override == "show_largest_single_record" and not is_duplicate_focus:
        vendor = resolved_context.get("vendor_name", "the vendor")
        rewritten = f"Show the single largest invoice from {vendor} ordered by amount descending limit 1"
        logger.info(f"Intent override applied: '{question}' → '{rewritten}'")
        return rewritten

    # ── If in duplicate focus, block show_largest and handle directly ──
    if is_duplicate_focus and intent_override == "show_largest_single_record":
        logger.info("Blocked show_largest_single_record — duplicate focus active")
        # Fall through to LLM rewriting with duplicate context

    # Skip rewriting for standalone questions
    if any(p in q_lower for p in STANDALONE_PATTERNS):
        logger.info("Standalone question — no rewrite needed")
        return question

    # Skip if no history
    if not history:
        return question

    needs_rewrite = any(ind in q_lower for ind in VAGUE_INDICATORS)

    if not needs_rewrite:
        return question

    try:
        context_str = json.dumps(resolved_context) if resolved_context else "{}"

        chain = REWRITE_PROMPT | llm
        result = chain.invoke({
            "question": question,
            "history": history,
            "last_results": str(last_metadata.get("results_sample", []))[:400],
            "resolved_context": context_str
        })
        rewritten = result.content.strip()

        if not rewritten or len(rewritten) < 5:
            return question

        logger.info(f"Rewritten: '{question}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return question