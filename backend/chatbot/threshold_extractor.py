import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from chatbot.llm_provider import invoke_with_fallback

logger = logging.getLogger(__name__)

THRESHOLD_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are a numeric threshold extractor. Determine if the question contains
a monetary threshold/filter condition (a minimum or maximum amount),
regardless of how it's phrased — any synonym for "above", "below",
"exceeding", "at least", "no more than", "north of", etc.

QUESTION: {question}

Return ONLY valid JSON:
{{
  "has_threshold": true or false,
  "operator": ">" | "<" | ">=" | "<=" | null,
  "value": <number or null>,
  "currency": "<3-letter code or null>"
}}

Examples:
"Show invoices above 10000 EUR" → {{"has_threshold": true, "operator": ">", "value": 10000, "currency": "EUR"}}
"invoices exceeding 50000" → {{"has_threshold": true, "operator": ">", "value": 50000, "currency": null}}
"bills under 500 USD" → {{"has_threshold": true, "operator": "<", "value": 500, "currency": "USD"}}
"at least 2000" → {{"has_threshold": true, "operator": ">=", "value": 2000, "currency": null}}
"no more than 1000 GBP" → {{"has_threshold": true, "operator": "<=", "value": 1000, "currency": "GBP"}}
"what are my biggest bills" → {{"has_threshold": false, "operator": null, "value": null, "currency": null}}
"second highest invoice" → {{"has_threshold": false, "operator": null, "value": null, "currency": null}}

Return ONLY JSON. No explanation. No markdown.
""")


def extract_amount_threshold(question: str) -> dict:
    """
    Semantically extract a monetary threshold from any phrasing —
    no keyword list, works for any synonym of above/below/exceeding/etc.
    """
    try:
        result = invoke_with_fallback(
            lambda llm: THRESHOLD_EXTRACTION_PROMPT | llm,
            {"question": question}
        )
        content = result.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        logger.error(f"Threshold extraction failed: {e}")
        return {"has_threshold": False, "operator": None, "value": None, "currency": None}


def apply_amount_threshold(rows: list, threshold: dict,
                            amount_key="amount", currency_key="currency",
                            base_amount_key="base_amount") -> list:
    """
    Apply an extracted threshold to a list of rows — works regardless of
    which special intent handler is calling it, and regardless of whether
    classification was "correct." This is what makes the answer honest
    even when intent detection guesses wrong.
    """
    if not threshold.get("has_threshold"):
        return rows

    op = threshold.get("operator")
    value = threshold.get("value")
    currency = threshold.get("currency")
    if op is None or value is None:
        return rows

    comparators = {
        ">": lambda a: a > value,
        "<": lambda a: a < value,
        ">=": lambda a: a >= value,
        "<=": lambda a: a <= value,
    }
    cmp = comparators.get(op)
    if not cmp:
        return rows

    filtered = []
    for row in rows:
        try:
            if currency:
                if row.get(currency_key) != currency:
                    continue
                amt = float(row.get(amount_key, 0))
            else:
                # No currency specified — compare against the fair,
                # converted EUR-equivalent value
                amt = float(row.get(base_amount_key, row.get(amount_key, 0)))
        except (TypeError, ValueError):
            continue
        if cmp(amt):
            filtered.append(row)
    return filtered