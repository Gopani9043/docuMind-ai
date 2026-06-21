import os
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from services.vendor_matcher import normalize
from chatbot.llm_provider import invoke_with_fallback
load_dotenv()
logger = logging.getLogger(__name__)


# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
#     temperature=0,
# )

CONTEXT_PROMPT = ChatPromptTemplate.from_template("""
You are a context resolver for a document analytics chatbot.

CONVERSATION HISTORY:
{history}

LAST QUERY TYPE:
{last_query_type}

LAST VISIBLE RESULTS (what user saw):
{last_results}

CURRENT CONVERSATIONAL FOCUS:
{conversation_focus}

CURRENT QUESTION:
{question}

CONTEXT RESOLUTION PRIORITY — follow this order strictly:

1. CONVERSATIONAL FOCUS FIRST
   If conversation_focus has a value, the follow-up is about THAT topic.
   Example: focus=duplicate_invoices → "how many times" means duplicate count
   Example: focus=vendor_analytics → "which one" means top vendor
   NEVER switch focus unless user explicitly introduces a new topic with new keywords.

2. LAST VISIBLE RESULTS SECOND
   If user says "that one", "the second one", "which one is newest" →
   resolve from last_results list, not from a new analytics query.
   Extract vendor, amount, filename directly from last_results — never guess.

3. FOLLOW-UP PATTERNS — map these exactly:
   "how many times" + focus=duplicate_invoices → intent_override=repetition_count
   "how many times" + focus=anything_else → count of results
   "show them" + focus=duplicates → intent_override=duplicate_followup
   "which one is newest/largest/smallest" → intent_override=list_navigation
   "the second one" / "third one" → intent_override=list_navigation
   "show only that vendor" → filter by vendor from last_results[0]
   "what currency" → currency from last_results[0]
   "when was it uploaded" → intent_override=list_navigation

4. INTENT OVERRIDE — only set these when 100% certain:
   show_largest_single_record → ONLY when user asks for single highest invoice
                                 AND conversation_focus is NOT duplicate_invoices
                                 AND conversation_focus is NOT a list topic
   duplicate_followup → when focus=duplicate_invoices and user wants to see them
   list_navigation → when user navigates previous list by position or attribute
   repetition_count → ONLY when focus=duplicate_invoices AND user asks "how many times"

5. CRITICAL — NEVER do these:
   - NEVER set intent_override=show_largest_single_record when focus=duplicate_invoices
   - NEVER switch to amount/vendor ranking when previous query was about duplicates
   - NEVER hallucinate vendor names — only use names from last_results or history
   - NEVER set vendor_name to a hardcoded value — extract from last_results only

6. VENDOR EXTRACTION RULE:
   If you need vendor_name, extract it from last_results[0].vendor or last_results[0].vendor_name
   If last_results is empty, extract from conversation history
   Never invent or assume a vendor name

Extract and return JSON:
{{
  "document_type": "invoice|contract|receipt|report|null",
  "vendor_name": "extracted from results or history — never invented",
  "currency": "extracted from results or null",
  "amount_reference": "specific numeric amount or null",
  "time_period": "this month|last month|this week|this year|null",
  "comparison_requested": true or false,
  "previous_query_type": "duplicate_detection|aggregation|list|single_record|null",
  "intent_override": "show_largest_single_record|duplicate_followup|list_navigation|repetition_count|null",
  "conversation_focus": "current topic of conversation — preserved from focus unless new topic introduced",
  "resolved_references": "what vague terms refer to — name the actual entity from results"
}}

Return ONLY valid JSON. No explanation. No markdown.
""")


# ── Vague terms that trigger LLM resolution ───────────────────────────────────
VAGUE_TERMS = [
    "this", "that", "those", "it", "them", "they",
    "the highest", "the lowest", "the largest", "the smallest",
    "same", "similar", "previous", "above", "that average",
    "closest", "nearest", "compared", "the one",
    # ── follow-up navigation ──
    "how many times", "repeat", "how often",
    "the second", "the third", "the fourth", "the last", "the first",
    "show them", "list them", "newest", "oldest", "latest",
    "what currency", "when was it", "which one", "that vendor",
    "that invoice", "that contract", "show me that", "show me the"
]


def resolve_context(
    question: str,
    history: str,
    last_metadata: dict,
    last_results: list = None,
    conversation_focus: dict = None
) -> dict:
    """
    Extract context clues from conversation history and memory.
    Fully universal — no hardcoded vendor names or values.
    All entities extracted from actual query results.

    Args:
        question: current user question
        history: recent conversation turns as string
        last_metadata: metadata from last assistant turn
        last_results: last visible results from Redis (universal)
        conversation_focus: current conversational topic from Redis (universal)
    """
    # ── Default context ───────────────────────────────────────────────────────
    default_context = {
        "document_type": None,
        "vendor_name": None,
        "currency": None,
        "amount_reference": None,
        "time_period": None,
        "comparison_requested": False,
        "previous_query_type": None,
        "intent_override": None,
        "conversation_focus": None,
        "resolved_references": None
    }

    # ── Bare/contentless command — never inherit any context ──────────────────
    # "Show", "List", "Get" alone have zero referential content; inheriting
    # vendor/currency/amount from stale results causes hallucinated transforms.
    bare_command_words = {"show", "list", "display", "get", "find", "fetch", "give"}
    q_tokens = [t for t in question.lower().strip("?.,! ").split() if t]
    if len(q_tokens) <= 1 and (not q_tokens or q_tokens[0] in bare_command_words):
        logger.info(f"Bare command '{question}' — returning clean context, no inheritance")
        return default_context


    # ── Fast rule-based resolution — no LLM needed ───────────────────────────
    q_lower = question.lower()

    # Time period detection
    if "this month" in q_lower:
        default_context["time_period"] = "this month"
    elif "last month" in q_lower:
        default_context["time_period"] = "last month"
    elif "this week" in q_lower:
        default_context["time_period"] = "this week"
    elif "this year" in q_lower:
        default_context["time_period"] = "this year"
    elif "last 30 days" in q_lower:
        default_context["time_period"] = "last 30 days"

    # Comparison detection
    if any(w in q_lower for w in ["compare", " vs ", "versus", "last month", "previous month"]):
        default_context["comparison_requested"] = True

    # ── Extract entities from actual results — universal ──────────────────────
    # Priority: last_results > last_metadata results_sample
    # Never hardcode — always extract from real data
    actual_results = last_results or last_metadata.get("results_sample", []) if last_metadata else []

    if actual_results:
        first = actual_results[0]
        raw_vendor = first.get("vendor") or first.get("vendor_name")
        if raw_vendor:
            default_context["vendor_name"] = raw_vendor
        if first.get("currency"):
            default_context["currency"] = first["currency"]
        if first.get("amount"):
            default_context["amount_reference"] = str(first["amount"])

    # ── Carry forward active filters from conversation focus ──────────────────
    # This ensures "smallest one" after "above 10000 EUR" keeps all filters
    if conversation_focus:
        if not default_context["vendor_name"] and conversation_focus.get("vendor"):
            default_context["vendor_name"] = conversation_focus["vendor"]
        if not default_context["currency"] and conversation_focus.get("currency"):
            default_context["currency"] = conversation_focus["currency"]
        # Carry forward amount threshold only for filter queries, not absolute amounts
        if conversation_focus.get("amount_threshold"):
            default_context["amount_threshold"] = conversation_focus["amount_threshold"]

    # ── Fast path: repetition count — no LLM needed ──────────────────────────
    # Universal: works for any duplicate group, any vendor
    if any(k in q_lower for k in ["how many times", "how often", "repeat", "how many time", "times does", "time it"]):
        focus_topic = conversation_focus.get("topic", "") if conversation_focus else ""
        if "duplicate" in focus_topic:
            default_context["intent_override"] = "repetition_count"
            default_context["previous_query_type"] = "duplicate_detection"
            logger.info("Fast path: repetition_count detected")
            return default_context

    # ── Fast path: list navigation — no LLM needed ───────────────────────────
    navigation_terms = ["the second", "the third", "the fourth", "the fifth", "the last", "the first"]

    # Never use list_navigation if question has filter keywords — it is a new query
    q_lower = question.lower().strip()
    # ── Pre-compute once — used in multiple places ──
    has_filter_keywords = any(k in q_lower for k in [
        "above ", "below ", "more than", "less than",
        "this month", "last month", "show invoices",
        "show contracts", "find invoices", "list invoices",
        "show me all", "list all", "show all",
        "compare the", "compare top", "top two", "top three",
        "risky", "suspicious ones", "only the", "show only",
        "related invoices", "related contracts",
        "convert", "mentally",
        "smallest", "largest", "cheapest", "most expensive",
        "highest amount", "lowest amount", "newest", "oldest",
        "latest", "earliest", "most recent",
        "which currency", "highest currency", "currency total",
        "total eur", "total usd", "total spending",
        "which month", "highest month", "month total"
    ])

    if any(term in q_lower for term in navigation_terms) and actual_results and not has_filter_keywords:
        default_context["intent_override"] = "list_navigation"
        logger.info("Fast path: list_navigation detected")
        return default_context

    # ── Skip LLM if no history and no vague terms ─────────────────────────────
    if not history:
        return default_context

    needs_llm = any(term in q_lower for term in VAGUE_TERMS)
    if not needs_llm:
        return default_context

    # ── Fast path: question contains a proper noun entity — always fresh query ─
    # Detects "Does CloudPeak appear...", "Is BrightPath in...", etc.
    # A proper noun sequence = 1+ consecutive capitalized words not at sentence start
    words = question.split()
    proper_nouns = []
    for i, w in enumerate(words):
        clean = w.strip("?.,!:;")
        if i > 0 and clean and clean[0].isupper() and len(clean) > 2:
            proper_nouns.append(clean)

    if proper_nouns:
        # Check if any proper noun looks like a vendor (not a common word)
        common_caps = {"Does", "Is", "Are", "Which", "What", "Show", "Find",
                       "Compare", "Who", "When", "Where", "How", "Can", "Do",
                       "Have", "Has", "The", "This", "That", "These", "Those"}
        vendor_like = [w for w in proper_nouns if w not in common_caps]

        # Also catch lowercase potential names (e.g. "dhruv", "tesla")
        q_words = question.strip("?.,!").split()
        stopwords_lower = {"does", "is", "are", "which", "what", "show", "find",
                          "appear", "in", "contracts", "invoices", "too", "also",
                          "the", "a", "an", "and", "or", "not", "any", "have",
                          "has", "do", "can", "where", "when", "how", "both"}
        for i, w in enumerate(q_words):
            clean = w.strip("?.,!:;").lower()
            if i > 0 and clean not in stopwords_lower and len(clean) > 2:
                if not clean[0].isupper():  # lowercase word
                    vendor_like.append(w)

        if vendor_like:
            logger.info(f"Fast path: proper noun entity detected {vendor_like} — fresh query")
            # Clear ALL inherited context — this is a fresh entity question
            return {
                "document_type": None,
                "vendor_name": None,
                "currency": None,
                "amount_reference": None,
                "time_period": None,
                "comparison_requested": False,
                "previous_query_type": None,
                "intent_override": None,
                "conversation_focus": None,
                "resolved_references": None
            }
    # ── Fast path: fresh analytical queries — never list_navigation ──────────
    # These always need a new SQL query, never navigate a cached list
    fresh_query_patterns = [
        "show the", "show me the", "list the", "find the", "get the",
        "most expensive", "cheapest", "lowest", "highest amount",
        "newest", "oldest", "latest", "earliest", "most recent",
        "top ", "bottom ", "first ", "last ", "all invoices",
        "all contracts", "all receipts", "10 most", "5 most",
        "10 cheapest", "5 cheapest", "10 largest", "5 largest",
    ]
    is_fresh_query = any(p in q_lower for p in fresh_query_patterns)

    # If it looks like a fresh analytical query and not a positional reference
    # ("the second one", "the third item"), skip LLM and return clean context
    positional_terms = ["the second", "the third", "the fourth", "the fifth", "item "]
    is_positional = any(p in q_lower for p in positional_terms)

    if is_fresh_query and not is_positional:
        logger.info("Fast path: fresh query detected — skipping LLM context resolution")
        return default_context

    # ── LLM resolution for complex vague references ───────────────────────────
    try:
        last_query_type = last_metadata.get("intent", "null") if last_metadata else "null"

        # Pass last_results as JSON — universal, contains real data
        results_str = json.dumps(actual_results[:5], default=str) if actual_results else "[]"

        # Pass conversation focus — universal topic tracking
        focus_str = json.dumps(conversation_focus) if conversation_focus else "{}"

        content = invoke_with_fallback(
            lambda llm: CONTEXT_PROMPT | llm,
            {
                "question": question,
                "history": history,
                "last_query_type": last_query_type,
                "last_results": results_str,
                "conversation_focus": focus_str
            }
        )

        # Clean markdown fences if present
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        resolved = json.loads(content)

        # ── Safety check: never allow hardcoded or invented vendors ──
        # If LLM returns a vendor not in actual results, discard it
        if resolved.get("vendor_name") and actual_results:
            known_vendors = [
                r.get("vendor") or r.get("vendor_name", "")
                for r in actual_results
            ]
            llm_vendor = resolved.get("vendor_name", "")
            # Check if LLM vendor roughly matches a known one
            vendor_valid = any(
                llm_vendor.lower() in v.lower() or v.lower() in llm_vendor.lower()
                for v in known_vendors if v
            )
            if not vendor_valid and known_vendors:
                # Use vendor from actual results instead
                resolved["vendor_name"] = default_context.get("vendor_name")

        # Never set list_navigation for filter queries
        has_filter_keywords = any(k in q_lower for k in [
            "above ", "below ", "more than", "less than",
            "this month", "last month", "show invoices",
            "show contracts", "find invoices", "list invoices",
            "show me all", "list all", "show all",
            "compare the", "compare top", "top two", "top three",
            "risky", "suspicious ones", "only the", "show only",
            "related invoices", "related contracts",
            "convert", "mentally","which currency", "currency has", "highest currency",
            "currency total", "by currency", "currency breakdown",
        ])
        # Block list_navigation for filter queries
        if has_filter_keywords and resolved.get("intent_override") == "list_navigation":
            resolved["intent_override"] = None
            logger.info("Safety: cleared list_navigation override for filter query")

        # Block list_navigation for ordinal+superlative — always needs fresh SQL
        ordinal_superlative = [
            "second largest", "second biggest", "second highest",
            "second smallest", "second cheapest", "second lowest",
            "second newest", "second oldest", "second most",
            "third largest", "third highest", "third smallest",
            "third cheapest", "fourth largest", "fifth largest",
        ]
        if any(p in q_lower for p in ordinal_superlative):
            resolved["intent_override"] = None
            logger.info("Safety: cleared list_navigation for ordinal+superlative — needs fresh SQL")

        if any(p in q_lower for p in ordinal_superlative):
            resolved["intent_override"] = None
            logger.info("Safety: cleared list_navigation for ordinal+superlative — needs fresh SQL")

        # ── ADD HERE ──
        # "when was it uploaded" → needs fresh SQL with created_at, never list_navigation
        upload_time_patterns = [
            "when was it uploaded", "when was it added", "upload date",
            "when was this uploaded", "upload time", "when did it arrive"
        ]
        if any(p in q_lower for p in upload_time_patterns):
            resolved["intent_override"] = None
            logger.info("Safety: cleared list_navigation for upload time query — needs fresh SQL")


        # ── Safety check: never override duplicate focus with amount ranking ──
        focus_topic = conversation_focus.get("topic", "") if conversation_focus else ""
        if "duplicate" in focus_topic:
            if resolved.get("intent_override") == "show_largest_single_record":
                resolved["intent_override"] = "duplicate_followup"
                logger.info("Safety: prevented show_largest_single_record override on duplicate focus")

        default_context.update(resolved)
        logger.info(f"Resolved context: {default_context}")

    except Exception as e:
        logger.error(f"Context resolution failed: {e}")

    return default_context