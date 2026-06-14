import os
import json
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

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are a financial analytics assistant.

Question:
{question}

Intent:
{intent}

Results:
{results}

Your job is to analyze the result structure and produce a clean business answer.

────────────────────────────────────
TYPE DETECTION RULES
────────────────────────────────────

TYPE: EMPTY
If Results is empty:
"No matching records found."

────────────────────────────────────

TYPE: SINGLE FACT
Conditions:
- Exactly 1 row
- No filename column

Output:
One concise sentence containing the answer.

Examples:
"BrightPath Analytics has the highest total spending at 959,452.64."
"The average invoice amount is 48,096.94."
"USD has the highest total invoice value at 787,696.70."
"The highest invoice month is March 2026 with 468,804.05."

────────────────────────────────────

TYPE: DOCUMENT LIST
Conditions:
- filename exists

Output:
Numbered list.

Format:

1. filename | vendor | amount currency
2. filename | vendor | amount currency

Optional fields:
- due_date
- issue_date
- end_date
- invoice_number

Examples:

1. invoice_203.pdf | BrightPath Analytics | 129900 EUR
2. invoice_045.pdf | Nordic Systems | 75798.98 EUR

If due_date exists:

1. invoice_203.pdf | BrightPath Analytics | 129900 EUR | due: 2026-07-15

End with:
"N results total."

Never show more than 10 rows.

────────────────────────────────────

TYPE: RANKING
Conditions:
- Ordered results
- Top vendors
- Largest invoices
- Smallest invoices

Output:

🏆 Top Results

1. Vendor A — 959,452.64
2. Vendor B — 496,783.43
3. Vendor C — 460,733.25

"N results total."

────────────────────────────────────

TYPE: GROUPED BREAKDOWN
Conditions:
- Multiple rows
- Aggregated values
- Group by vendor, currency, document type, month, etc.

Output:

🔹 Group Name: Value
🔹 Group Name: Value
🔹 Group Name: Value

Example:

🔹 BrightPath Analytics: 959,452.64
🔹 FinEdge Consulting: 496,783.43
🔹 EuroData AG: 460,733.25

────────────────────────────────────

TYPE: COMPARISON
Conditions:
- Exactly 2 or more grouped rows
- User asks compare, versus, vs

Output:

Compare all returned groups.

Example:

💱 EUR: 493,503.78
💱 USD: 787,696.70

Winner: USD leads by 294,192.92.

Example:

🏢 BrightPath Analytics: 959,452.64
🏢 FinEdge Consulting: 496,783.43

Winner: BrightPath Analytics leads by 462,669.21.

Example:

📅 This Month: 18 invoices | 245,000
📅 Last Month: 15 invoices | 198,000

Difference: +3 invoices | +47,000 spending.

Never assume currencies, vendors, months, or periods.
Use whatever groups are present in Results.

────────────────────────────────────

TYPE: TREND
Conditions:
- month column exists
OR
date-based aggregation

Output:

📈 Trend

Mar 2026: 468,804.05
Apr 2026: 392,100.50
May 2026: 410,000.00

Insight:
One short sentence describing the trend.

Examples:
"Spending increased over the period."
"Spending remained stable."
"Spending peaked in March."

────────────────────────────────────

TYPE: DUPLICATES
Conditions:
- repeat_count exists

Output:

INV-2024-002 appears 2 times.

If multiple:

INV-2024-002 appears 2 times.
INV-2024-017 appears 3 times.

────────────────────────────────────

TYPE: CONTRACT EXPIRY
Conditions:
- end_date exists

Output:

📄 Contract Name | Vendor | expires: YYYY-MM-DD

End with:
"N contracts found."

────────────────────────────────────

GENERAL RULES
────────────────────────────────────

- Never mention SQL.
- Never mention databases.
- Never repeat the question.
- Never say "Based on the data".
- Never say "I understood".
- Never explain how results were generated.
- Never invent values.
- Use exactly the numbers present in Results.
- Preserve currencies when present.
- Preserve dates when present.
- If count exists, show it.
- If count does not exist, never invent counts.
- Use concise business language.
- Maximum 10 displayed rows.
- Always choose the output type from the actual result structure, not from the question text.

Answer:
""")

CLARIFICATION_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful document analytics assistant.
The user wants clarification about your previous answer.
Explain it clearly and simply.

CONVERSATION HISTORY:
{history}

CLARIFICATION REQUEST:
{question}

Response:
""")


def synthesize_response(
    original_question: str,
    rewritten_question: str,
    results: list,
    history: str,
    intent: str
) -> str:
    """Synthesize response — lists generated in Python, facts by LLM."""
    try:
        if intent == "greeting":
            chain = GREETING_PROMPT | llm
            response = chain.invoke({
                "question": original_question,
                "history": history or "No history"
            })
            return response.content.strip()

        if intent == "clarification":
            chain = CLARIFICATION_PROMPT | llm
            response = chain.invoke({
                "question": original_question,
                "history": history or "No history"
            })
            return response.content.strip()

        if not results:
            q = original_question.lower()
            if any(k in q for k in ["failed", "error", "broken"]):
                return "No failed documents found. All documents processed successfully."
            if any(k in q for k in ["overdue", "past due", "late"]):
                return "No overdue invoices found. All invoices are within their due dates."
            if any(k in q for k in ["expiring", "expired", "ending soon"]):
                return "No contracts expiring in the next 30 days."
            if any(k in q for k in ["receipt", "kassenbon"]):
                return "No receipts found in your documents."
            return "No results found."

        # ── Period/month comparisons: handle in Python, bypass the LLM ──
        has_period = results and results[0].get("period")
        has_month = results and results[0].get("month")
        looks_like_comparison = (
            ("compare" in original_question.lower() or " vs " in original_question.lower() or "versus" in original_question.lower())
            and not results[0].get("filename")
            and len(results) <= 10
        )
        if has_period or has_month or looks_like_comparison:
            return _build_list_response(results, original_question)

        # ── Detect query type ──
        q = original_question.lower()
        is_aggregation = any(k in q for k in [
            "total", "sum", "average", "how much", "how many",
            "top vendor", "which vendor", "highest", "lowest",
            "compare", "percentage", "breakdown"
        ]) and len(results) <= 5

        has_growth = results and results[0].get("growth_percent") is not None
        if has_period or has_month or has_growth or looks_like_comparison:
            return _build_list_response(results, original_question)

        # Single fact only for aggregation results, not for list results with 1 item
        is_single_fact = len(results) == 1 and not results[0].get("filename")

        if is_aggregation or is_single_fact:
            # Use LLM for aggregation and single facts
            results_str = json.dumps(results[:10], indent=2, default=str)
            chain = SYNTHESIS_PROMPT | llm
            response = chain.invoke({
                "question": original_question,
                "results": results_str,
                "intent": intent
            })
            return response.content.strip()

        # ── List query — generate in Python, no LLM ──
        return _build_list_response(results, original_question)

    except Exception as e:
        logger.error(f"Response synthesis failed: {e}")
        return _build_list_response(results, original_question) if results else "No results found."


def _build_list_response(results: list, question: str) -> str:
    """
    Build list response in pure Python.
    Guarantees no duplicates, correct count, correct format.
    """

    # ── Special case: aggregation results without filename ──
    # GROUP BY queries return rows with no filename field
    # ── Special case: aggregation results without filename ──
    if results and not results[0].get("filename"):
        # Check if this is a comparison result (has 'period' field)
        if results[0].get("period"):
            lines = []
            for row in results:
                period = row.get("period", "")
                count = row.get("count", 0)
                total = row.get("total", "")
                try:
                    total_fmt = f"{float(total):,.2f}" if total else "N/A"
                except (ValueError, TypeError):
                    total_fmt = str(total)
                lines.append(
                    f"📊 {period}: {count} invoices | "
                    f"Total: {total_fmt} (mixed currencies)"
                )

            if len(results) == 2:
                try:
                    c1 = int(results[0].get("count", 0))
                    c2 = int(results[1].get("count", 0))
                    t1 = float(results[0].get("total", 0) or 0)
                    t2 = float(results[1].get("total", 0) or 0)
                    count_diff = abs(c1 - c2)
                    total_diff = abs(t1 - t2)
                    more = results[0].get("period") if c1 > c2 else results[1].get("period")
                    lines.append(
                        f"\n{more} had more activity: "
                        f"+{count_diff} invoices, "
                        f"+{total_diff:,.2f} total value (mixed currencies)"
                    )
                except (ValueError, TypeError):
                    pass

            return "\n".join(lines)

            # Add comparison summary
            if len(results) == 2:
                r1 = results[0]
                r2 = results[1]
                diff = abs(int(r1.get("count", 0)) - int(r2.get("count", 0)))
                winner = r1.get("period") if int(r1.get("count", 0)) > int(r2.get("count", 0)) else r2.get("period")
                lines.append(f"\n{winner} had more invoices by {diff}.")

            return "\n".join(lines)

        # ── Growth-between-periods result ──
        if results[0].get("growth_percent") is not None and results[0].get("month"):
            row = results[0]
            label = row.get("vendor") or row.get("currency") or "Result"
            try:
                growth = float(row.get("growth_percent", 0))
            except (ValueError, TypeError):
                growth = 0.0
            prev_month = row.get("prev_month", "")
            month = row.get("month", "")
            try:
                prev_total = float(row.get("prev_total", 0) or 0)
            except (ValueError, TypeError):
                prev_total = 0.0
            try:
                total = float(row.get("total", 0) or 0)
            except (ValueError, TypeError):
                total = 0.0
            def fmt_month(m):
                if not m:
                    return ""
                return str(m)[:7]  # YYYY-MM

            prev_total_fmt = f"{prev_total:,.2f}"
            total_fmt = f"{total:,.2f}"
            direction = "grew" if growth >= 0 else "declined"
            return (
                f"📈 {label} {direction} {abs(growth):,.2f}% "
                f"from {fmt_month(prev_month)} ({prev_total_fmt}) "
                f"to {fmt_month(month)} ({total_fmt})."
            )

        # ── Vendor/group comparison (e.g. "X vs Y total spending") ──
        if len(results) >= 2 and any(
            k in row for row in results for k in ("vendor", "total_spent", "total_amount", "total")
        ):
            label_key = next((k for k in results[0] if k not in (
                "total_spent", "total_amount", "total", "count",
                "total_spending", "total_paid"
            )), None)
            value_key = next((k for k in results[0]
                               if ("total" in k.lower() or "amount" in k.lower() or "paid" in k.lower())
                               and k != "vendor"), None)

            if label_key and value_key:
                # Merge rows whose vendor name normalizes to the same entity
                # (handles "BrightPath Analytics" vs "BrightPath Analytics Ltd")
                merged = {}
                for row in results:
                    label = row.get(label_key, "Unknown")
                    raw = row.get(value_key, 0)
                    try:
                        val = float(raw)
                    except (ValueError, TypeError):
                        val = 0.0
                    norm = normalize(label) or label.lower()
                    if norm in merged:
                        merged[norm]["value"] += val
                        # prefer the shorter/cleaner-looking label
                        if len(label) < len(merged[norm]["label"]):
                            merged[norm]["label"] = label
                    else:
                        merged[norm] = {"label": label, "value": val}

                parsed = [(v["label"], v["value"]) for v in merged.values()]
                parsed.sort(key=lambda x: x[1], reverse=True)

                lines = [f"🏢 {label}: {val:,.2f}" for label, val in parsed]

                if len(parsed) >= 2:
                    top, second = parsed[0], parsed[1]
                    diff = top[1] - second[1]
                    lines.append(f"\nWinner: {top[0]} leads by {diff:,.2f}.")

                lines.append(f"\n{len(parsed)} result{'s' if len(parsed) != 1 else ''} total.")
                return "\n".join(lines)

        # Generic aggregation — format as key: value pairs
        lines = []
        for i, row in enumerate(results, 1):
            parts = []
            for k, v in row.items():
                if v is not None and str(v).strip():
                    try:
                        if isinstance(v, (int, float)) or str(v).replace('.','').isdigit():
                            parts.append(f"{k.replace('_', ' ').title()}: {float(v):,.2f}")
                        else:
                            parts.append(f"{k.replace('_', ' ').title()}: {v}")
                    except (ValueError, TypeError):
                        parts.append(f"{k.replace('_', ' ').title()}: {v}")
            lines.append(f"{i}. {' | '.join(parts)}")
        total = len(results)
        return "\n".join(lines) + f"\n{total} result{'s' if total != 1 else ''} total."
    # ── Regular list with filenames ──
    seen = set()
    unique = []
    for row in results:
        filename = row.get("filename", "")
        if filename:
            if filename not in seen:
                seen.add(filename)
                unique.append(row)
        else:
            unique.append(row)

    lines = []
    show_count = min(len(unique), 50)

    for i, row in enumerate(unique[:show_count], 1):
        filename = row.get("filename", "")
        vendor = (
            row.get("vendor") or
            row.get("vendor_name") or
            row.get("parties") or ""
        )
        amount = (
            row.get("amount") or
            row.get("total_paid") or
            row.get("value") or
            row.get("total") or ""
        )
        currency = row.get("currency", "")
        repeat_count = row.get("repeat_count")
        invoice_number = row.get("invoice_number", "")
        due_date = row.get("due_date") or row.get("end_date") or ""

        # Build exactly ONE line per row
        if repeat_count:
            label = invoice_number or filename
            lines.append(f"{i}. {label} | {vendor} | {repeat_count} times")
        elif filename and vendor and amount and due_date:
            lines.append(f"{i}. {filename} | {vendor} | {amount} {currency} | due: {due_date}".strip())
        elif filename and vendor and amount:
            lines.append(f"{i}. {filename} | {vendor} | {amount} {currency}".strip())
        elif filename and vendor and due_date:
            lines.append(f"{i}. {filename} | {vendor} | due: {due_date}")
        elif filename and vendor:
            lines.append(f"{i}. {filename} | {vendor}")
        elif vendor and amount:
            lines.append(f"{i}. {vendor} | {amount} {currency}".strip())
        else:
            lines.append(f"{i}. {filename or vendor or 'unknown'}")
    result_text = "\n".join(lines)

    total = len(unique)
    if total > show_count:
        result_text += f"\n... and {total - show_count} more results. {total} total."
    else:
        result_text += f"\n{total} result{'s' if total != 1 else ''} total."

    return result_text



def synthesize_error(question: str, error: str) -> str:
    """Generate a helpful error message."""
    return "I could not process that query. Try rephrasing — for example: 'Show all invoices' or 'Top vendors by amount'."


def synthesize_general(question: str, history: str) -> str:
    """Handle general questions."""
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for DocParse document processing system.
        Answer clearly and concisely. Stay focused on document processing topics.

        Conversation history:
        {history}

        Question: {question}
        Answer:
        """)
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "history": history or "No history"
        })
        return response.content.strip()
    except Exception as e:
        logger.error(f"General synthesis failed: {e}")
        return "I can help you analyze your documents. Try asking about invoices, contracts, or vendors."


def synthesize_multi(
    original_question: str,
    sub_results: list,
    history: str
) -> str:
    """Synthesize response from multiple sub-question results."""
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an executive financial analyst.
        Combine these multiple query results into one concise analytical response.

        RULES:
        - Lead with the most important insight
        - Max 5 sentences total
        - Include all key numbers with currency
        - Never say "Would you like..." or "I hope this helps"
        - Be direct and analytical

        ORIGINAL QUESTION: {question}
        SUB-RESULTS: {results}
        HISTORY: {history}

        Response:
        """)
        chain = prompt | llm
        response = chain.invoke({
            "question": original_question,
            "results": str(sub_results)[:2000],
            "history": history or "No history"
        })
        return response.content.strip()
    except Exception as e:
        logger.error(f"Multi synthesis failed: {e}")
        return "\n\n".join([r.get("answer", "") for r in sub_results if r.get("answer")])