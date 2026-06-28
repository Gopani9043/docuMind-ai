import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from chatbot.llm_provider import invoke_with_fallback

logger = logging.getLogger(__name__)

RANKING_SELECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a query parser for ranked-list questions (vendors, invoices,
contracts, months — any scored/ranked entity). Determine exactly what
slice of the ranking the user wants, what they're measuring by, and
whether they're asking about WHEN something occurred.

QUESTION: {question}

Return ONLY valid JSON:
{{
  "selection_type": "single_rank | top_n | bottom_n | tier | average | compare | full_list",
  "rank_position": <integer or null>,
  "n": <integer or null>,
  "tier": "high | medium | low | null",
  "compare_items": [<name fragments mentioned, or empty list>],
  "show_month": true or false,
  "metric": "amount | count",
  "scope": "invoice | contract"
}}

RULES:
- "most" / "highest" / "biggest" / "riskiest" (singular) → single_rank, rank_position=1
- "second most" / "second highest" / "second riskiest" → single_rank, rank_position=2
- "third most" → single_rank, rank_position=3
- "least" / "lowest" / "smallest" / "safest" (singular) → single_rank, rank_position=-1
- "second least" / "second smallest" / "second safest" → single_rank, rank_position=-2
- "top N" (N is a number) → top_n, n=N
- "bottom N" / "lowest N" / "smallest N" → bottom_n, n=N
- "medium" / "average" / "moderate" / "middle" / "closest to average" /
  "closest to the average" → average (a SINGLE item, the one nearest
  the mean — NOT a tier/category)
- "medium tier" / "moderate tier" / "categorize as medium" → tier, tier="medium"
  (a GROUP of items, distinct from the single-item "average" case above)
- "above average" / "high tier" → tier, tier="high"
- "below average" / "low tier" → tier, tier="low"
- "high/medium/low" categorization request → tier, tier=<high|medium|low>
- "compare X and Y" / "is X more/less than Y" → compare, compare_items=[X,Y]
- "rank all" / "show full ranking" / "list all by [metric]" → full_list
- show_month=true ONLY if the question explicitly asks WHEN something
  happened — "which month", "what month", "in which period", "when did
  this occur", "which quarter". Otherwise show_month=false.

CRITICAL EXCLUSION — MONETARY THRESHOLDS ARE NEVER RANK POSITIONS:
- If a number in the question is immediately preceded by "above",
  "below", "over", "under", "more than", "less than", "at least",
  "at most", "exceeding" — that number is a MONEY THRESHOLD, not a
  rank position or count. In this case return:
  {{"selection_type": "full_list", "rank_position": null, "n": null,
    "tier": null, "compare_items": [], "show_month": false, "metric": "amount"}}
- Only treat a number as rank_position/n when it's explicitly used as
  an ORDINAL ("second", "3rd") or as an explicit COUNT REQUEST phrased
  as "top N [items]" or "bottom N [items]" — e.g. "top 5 vendors".
- Examples:
  "Show invoices above 10000 EUR" → full_list (10000 is a threshold)
  "Top 10000 invoices" → top_n, n=10000 (explicit "top N" phrasing)
  "Second highest invoice" → single_rank, rank_position=2
  
METRIC RULE — decide by MEANING, not by keyword presence:
- metric="amount" when the question is fundamentally about financial
  VALUE — invoice size, spending, totals, biggest/highest/most expensive
  in a monetary sense. Example: "which month had the highest invoice" is
  about a money figure, even though it says "invoice" not "amount".
- metric="count" when the question is fundamentally about QUANTITY/HOW
  MANY documents exist — "how many invoices", "busiest month",
  "number of invoices", "most invoices received".
- Default to metric="amount" when genuinely ambiguous, since most
  business questions about "highest/biggest X" refer to value.

- scope="contract" when the question explicitly says "contract(s)",
  "agreement(s)", or asks about contract value/parties. Otherwise
  scope="invoice" (default).

Return ONLY JSON. No explanation. No markdown.
""")


def parse_ranking_selection(question: str) -> dict:
    """
    Generic ranking-position parser — reusable across ANY feature that
    shows a ranked list of scored entities (vendor importance, vendor
    risk, invoice size, contract value, etc). Returns structured intent
    instead of requiring every handler to maintain its own keyword list.
    """
    try:
        result = invoke_with_fallback(
            lambda llm: RANKING_SELECTION_PROMPT | llm,
            {"question": question}
        )
        content = result.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        logger.error(f"Ranking selection parsing failed: {e}")
        return {"selection_type": "single_rank", "rank_position": 1}


def select_from_ranking(ranked: list, selection: dict, format_line_fn):
    """
    Applies a parsed selection to an already-sorted list of (name, score)
    tuples, using format_line_fn(name, score, idx=None) -> str to render
    each line. Returns formatted lines + metadata.
    """
    n_items = len(ranked)
    sel_type = selection.get("selection_type", "single_rank")
    lines = []

    if sel_type == "single_rank":
        pos = selection.get("rank_position") or 1
        idx = (pos - 1) if pos > 0 else (n_items + pos)
        idx = max(0, min(idx, n_items - 1))
        name, score = ranked[idx]
        lines.append(format_line_fn(name, score))
        return {"lines": lines, "selected": [(name, score)], "position": pos}

    if sel_type in ("top_n", "bottom_n"):
        n = max(1, min(int(selection.get("n") or 3), n_items))
        subset = ranked[:n] if sel_type == "top_n" else list(reversed(ranked))[:n]
        for i, (name, score) in enumerate(subset, 1):
            lines.append(format_line_fn(name, score, idx=i))
        return {"lines": lines, "selected": subset}

    if sel_type == "average":
        avg_score = sum(s for _, s in ranked) / n_items
        closest = min(ranked, key=lambda x: abs(x[1] - avg_score))
        lines.append(format_line_fn(closest[0], closest[1]))
        return {"lines": lines, "selected": [closest], "average": avg_score}

    if sel_type == "tier":
        tier = selection.get("tier", "high")
        avg_score = sum(s for _, s in ranked) / n_items
        if tier == "high":
            subset = [(n, s) for n, s in ranked if s >= avg_score]
        elif tier == "low":
            subset = [(n, s) for n, s in ranked if s < avg_score]
        else:
            # Use a relative tolerance (15% of the average) instead of an
            # absolute number — works correctly whether scores are 0-100
            # importance points or six-figure currency totals.
            tolerance = abs(avg_score) * 0.15
            subset = [(n, s) for n, s in ranked if abs(s - avg_score) <= tolerance]
            if not subset and ranked:
                # Guarantee at least one result — fall back to the single
                # closest item rather than returning nothing.
                closest = min(ranked, key=lambda x: abs(x[1] - avg_score))
                subset = [closest]
        for i, (name, score) in enumerate(subset[:10], 1):
            lines.append(format_line_fn(name, score, idx=i))
        return {"lines": lines, "selected": subset, "average": avg_score}

    if sel_type == "compare":
        names = selection.get("compare_items", []) or []
        matched = []
        for target in names:
            for name, score in ranked:
                if target.lower() in name.lower():
                    matched.append((name, score))
                    break
        for name, score in matched:
            lines.append(format_line_fn(name, score))
        return {"lines": lines, "selected": matched}

    for i, (name, score) in enumerate(ranked[:10], 1):
        lines.append(format_line_fn(name, score, idx=i))
    return {"lines": lines, "selected": ranked[:10], "total": n_items}