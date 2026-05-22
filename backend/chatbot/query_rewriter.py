import os
import logging
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
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
Before applying any rewriting rules, check if resolved_context contains
"intent_override": "show_largest_single_record".

If intent_override is "show_largest_single_record":
- Rewrite to: "Show the single largest invoice from [vendor_name] ordered by amount descending limit 1"
- Always use the vendor_name from resolved_context
- Never rewrite as closest-to-average
- Never rewrite as a list query
- This overrides ALL other rules below

REWRITING RULES:
- "the highest one" → "show the invoice with highest amount from previous results"
- "those invoices" → "show invoices [with same filters as previous query]"
- "that vendor" → "show documents from [vendor name from previous result]"
- "compare with last month" → "compare [previous query subject] this month vs last month"
- "what about contracts" → "show same analysis but for contracts instead"
- "which ones are overdue" → "show overdue invoices [from previous filters]"
- "show me more" → "show more results from previous query"
- "that average" → "use the average value [X] from previous answer"
- "the closest one" → "find the invoice closest to [amount from previous answer]"
- "which invoice is this" + context has vendor → "Show the largest single invoice from [vendor] ordered by amount descending limit 1"
- "show me the invoice" after total query → "Show the largest single invoice from [vendor] ordered by amount descending limit 1"
- "which one" after aggregation → "Show the top individual record by amount descending limit 1"
- "when was it uploaded" + context has vendor → "Show filename and created_at for the largest invoice from [vendor]"
- "show related contracts" + context has vendor → "Show contracts where parties contains [vendor]"
- If question is already clear and specific → return it unchanged

Return ONLY the rewritten question. No explanation. No quotes. No markdown.

Rewritten question:
""")

# Questions that are clearly standalone — skip rewriting
STANDALONE_PATTERNS = [
    "show all", "list all", "how many total",
    "give me all", "find all", "what is the total",
    "top 10", "top 5", "show invoices", "show contracts",
    "show receipts", "show documents"
]

# Vague reference indicators — always trigger rewrite
VAGUE_INDICATORS = [
    "the highest", "the lowest", "that one", "those", "this one",
    "that invoice", "that vendor", "that contract", "the same",
    "compare", "what about", "which ones", "show me more",
    "that average", "closest", "nearest", "the best", "the worst",
    "previous", "above mentioned", "it ", "them ", "they ",
    # ── ADDED: follow-up question patterns ──
    "which invoice is this", "which one is this", "what is this",
    "show me the invoice", "show me that", "which one",
    "when was it", "when was it uploaded", "show related",
    "is this", "what invoice", "what contract"
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

    # ── ADDED: Always rewrite if intent_override is set ──
    intent_override = resolved_context.get("intent_override") if resolved_context else None
    if intent_override == "show_largest_single_record":
        vendor = resolved_context.get("vendor_name", "the vendor")
        rewritten = f"Show the single largest invoice from {vendor} ordered by amount descending limit 1"
        logger.info(f"Intent override applied: '{question}' → '{rewritten}'")
        return rewritten

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
        # ── FIXED: No truncation on resolved_context ──
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