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

CHECK IN THIS EXACT ORDER — stop at the first match:
1. EMPTY
2. TREND (month column exists in any row) — always wins over Comparison/Grouped Breakdown
3. DUPLICATES
4. CONTRACT EXPIRY
5. DOCUMENT LIST
6. COMPARISON
7. GROUPED BREAKDOWN
8. RANKING
9. SINGLE FACT

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
- Group by vendor, currency, document type, etc.
- Rows do NOT contain a "month" column (month data is always TREND, never Grouped Breakdown)

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
- Rows do NOT contain a "month" column (month data is always TREND, never Comparison)

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
Conditions (check this BEFORE Comparison/Grouped Breakdown — always wins if true):
- month column exists in any row
OR
date-based aggregation

Output:

📈 Trend

Mar 2026: 468,804.05
Apr 2026: 392,100.50
May 2026: 410,000.00

Insight:
One short sentence describing the trend.

Display rows in the exact order given in Results — never resort by value.
Never say "Winner" or "leads by" — that phrasing belongs only to COMPARISON.

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
- NEVER assume USD or any default currency symbol
- NEVER use "$" unless the data explicitly contains currency = "USD"
- If results have multiple currencies, ALWAYS show each currency separately — never combine into one number

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
        original_question = original_question or ""
        rewritten_question = rewritten_question or ""
        history = history or ""
        intent = intent or ""
    
        if intent == "greeting":
            return invoke_with_fallback(
                lambda llm: GREETING_PROMPT | llm,
                {
                    "question": original_question,
                    "history": history or "No history"
                }
            )

        if intent == "clarification":
            return invoke_with_fallback(
                lambda llm: CLARIFICATION_PROMPT | llm,
                {
                    "question": original_question,
                    "history": history or "No history"
                }
            )

        if not results:
            q = original_question.lower()
            if any(k in q for k in ["failed", "error", "broken"]):
                return "No failed documents found. All documents processed successfully."
            # Check combined overdue + expiring FIRST, before generic overdue check
            if any(k in q for k in ["overdue", "past due", "late"]) and \
               any(k in q for k in ["expiring", "expire", "contract"]):
                return (
                    "No overdue invoices found from vendors who also have "
                    "contracts expiring in the next 30 days. There may be "
                    "overdue invoices in the system, but none of those "
                    "vendors currently have expiring contracts."
                )
            if any(k in q for k in ["overdue", "past due", "late"]):
                return "No overdue invoices found. All invoices are within their due dates."
            if any(k in q for k in ["expiring", "expired", "ending soon"]):
                return "No contracts expiring in the next 30 days."
            if any(k in q for k in ["receipt", "kassenbon"]):
                return "No receipts found in your documents."
            return "No results found."

        # ── Anomaly queries: use LLM to explain reasons ──
        anomaly_words = ["unusual", "suspicious", "anomal", "outlier", "fraud", "weird", "abnormal"]
        if any(w in original_question.lower() for w in anomaly_words):
            return synthesize_anomaly(results, original_question, history)

        # ── Period/month comparisons: handle in Python, bypass the LLM ──
        has_period = results and "period" in results[0]
        has_month = results and "month" in results[0]
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
            return invoke_with_fallback(
                lambda llm: SYNTHESIS_PROMPT | llm,
                {
                    "question": original_question,
                    "results": results_str,
                    "intent": intent
                }
            )


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
        # ── Month trend result — MUST be checked before vendor comparison ──
        # (trend rows also have 'total_amount' which would otherwise match
        # the vendor comparison block below)
        if "month" in results[0]:
            from datetime import datetime
            lines = ["📈 Monthly Trend\n"]
            sorted_results = sorted(results, key=lambda r: r.get("month") or "")

            for row in sorted_results:
                month_raw = row.get("month", "")
                month_label = str(month_raw)[:7] if month_raw else "Unknown"
                try:
                    dt = datetime.fromisoformat(str(month_raw).replace("Z", "+00:00"))
                    month_label = dt.strftime("%b %Y")
                except Exception:
                    pass

                count = row.get("invoice_count") or row.get("count") or 0
                amount = row.get("total_amount") or row.get("amount") or row.get("total") or 0
                currency = row.get("currency", "")
                try:
                    amount_fmt = f"{float(amount):,.2f}"
                except (ValueError, TypeError):
                    amount_fmt = str(amount)

                lines.append(f"{month_label}: {count} invoices | {amount_fmt} {currency}".strip())

            try:
                amounts = [float(r.get("total_amount") or r.get("amount") or 0) for r in sorted_results]
                if len(amounts) >= 2:
                    if amounts[-1] > amounts[0]:
                        insight = "Spending increased over the period."
                    elif amounts[-1] < amounts[0]:
                        insight = "Spending decreased over the period."
                    else:
                        insight = "Spending remained stable."
                    lines.append(f"\n{insight}")
            except (ValueError, TypeError):
                pass

            lines.append(f"\n{len(results)} result{'s' if len(results) != 1 else ''} total.")
            return "\n".join(lines)

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
                    label = row.get(label_key) or "Unknown"
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

    def clean_parties(raw_parties):
        """Parse contract parties JSON into a clean readable name string."""
        if not raw_parties:
            return ""
        try:
            parsed = json.loads(raw_parties) if isinstance(raw_parties, str) else raw_parties
            if isinstance(parsed, list):
                names = [
                    p.get("name", "") if isinstance(p, dict) else str(p)
                    for p in parsed
                ]
                names = [
                    n for n in names
                    if n and n not in ("[MISSING_COMPANY_NAME]", "PARTY B (CLIENT)")
                ]
                return " / ".join(names)
            return str(parsed)
        except (json.JSONDecodeError, TypeError):
            return str(raw_parties)

    lines = []
    show_count = min(len(unique), 50)

    for i, row in enumerate(unique[:show_count], 1):
        filename = row.get("filename", "")
        is_contract_row = bool(row.get("parties")) and not row.get("vendor_name") and not row.get("vendor")
        vendor = (
            row.get("vendor") or
            row.get("vendor_name") or
            (clean_parties(row.get("parties")) if row.get("parties") else "")
        )
        amount = (
            row.get("amount") or
            row.get("total_paid") or
            row.get("value") or
            row.get("total") or 
            row.get("total_amount") or ""
        )
        currency = row.get("currency", "")
        repeat_count = row.get("repeat_count")
        invoice_number = row.get("invoice_number", "")
        due_date = row.get("due_date") or row.get("end_date") or ""
        date_label = "due" if row.get("due_date") else ("expires" if row.get("end_date") else "due")
        has_dup = row.get("has_duplicate", "")
        issue_date = row.get("issue_date", "")
        overdue_days = row.get("overdue_days", "")

        # Detect missing fields
        missing = []
        if not row.get("value") and not row.get("amount") and not row.get("total"):
            missing.append("value")
        if not row.get("due_date") and not row.get("end_date") and not row.get("start_date") and "date" in question.lower():
            missing.append("date")
        if not row.get("vendor") and not row.get("vendor_name") and not row.get("parties"):
            missing.append("vendor")
        if not row.get("currency"):
            missing.append("currency")
        missing_str = f" | ⚠️ missing: {', '.join(missing)}" if missing else ""

        if repeat_count:
            label = invoice_number or filename
            lines.append(f"{i}. {label} | {vendor} | {repeat_count} times")
        elif filename and vendor and amount and due_date:
            line = f"{i}. {filename} | {vendor} | {amount} {currency} | {date_label}: {due_date}"
            if overdue_days:
                line += f" | {overdue_days} days overdue"
            if has_dup == "Yes":
                line += " ⚠️ DUPLICATE"
            lines.append(line.strip())
        elif filename and vendor and amount:
            line = f"{i}. {filename} | {vendor} | {amount} {currency}{missing_str}".strip()
            if issue_date:
                line += f" | issued: {issue_date}"
            if has_dup == "Yes":
                line += " ⚠️ DUPLICATE"
            lines.append(line)
        elif filename and vendor and due_date:
            lines.append(f"{i}. {filename} | {vendor} | {date_label}: {due_date}{missing_str}")
        elif filename and vendor:
            lines.append(f"{i}. {filename} | {vendor}{missing_str}")
        elif filename and amount:
            lines.append(f"{i}. {filename} | {amount} {currency}{missing_str}".strip())
        elif vendor and amount:
            lines.append(f"{i}. {vendor} | {amount} {currency}{missing_str}".strip())
        else:
            lines.append(f"{i}. {filename or vendor or 'unknown'}{missing_str}")
    result_text = "\n".join(lines)

    total = len(unique)
    if total > show_count:
        result_text += f"\n... and {total - show_count} more results. {total} total."
    else:
        result_text += f"\n{total} result{'s' if total != 1 else ''} total."

    # Add duplicate summary if has_duplicate column present
    if unique and "has_duplicate" in unique[0]:
        dup_count = sum(1 for r in unique if r.get("has_duplicate") == "Yes")
        if dup_count == 0:
            result_text += "\n✅ No duplicate invoice numbers found among these results."
        else:
            result_text += f"\n⚠️ {dup_count} invoice(s) have duplicate invoice numbers."
        oldest = unique[0]
        oldest_vendor = oldest.get("vendor") or oldest.get("vendor_name", "")
        oldest_date = oldest.get("issue_date") or oldest.get("due_date") or "unknown"
        result_text += (
            f"\n📅 Oldest: {oldest.get('filename', 'unknown')} | "
            f"{oldest_vendor} | issued: {oldest_date}"
        )

    return result_text

def synthesize_anomaly(results: list, question: str, history: str) -> str:
    """Use LLM to explain WHY each result is anomalous."""
    try:
        prompt = ChatPromptTemplate.from_template("""
You are a financial fraud analyst reviewing invoice data.

The following invoices were flagged as statistically unusual or anomalous:
{results}

For each flagged invoice, explain in ONE sentence why it looks suspicious or unusual
(e.g. amount is far below/above average, duplicate detected, missing fields, etc).

Format:
⚠️ filename | vendor | amount — Reason: [one sentence explanation]

Be specific. Use the actual numbers. Never say "based on the data".
Answer:
""")
        return invoke_with_fallback(
            lambda llm: prompt | llm,
            {
                "results": json.dumps(results[:10], indent=2, default=str),
            }
        )
    except Exception as e:
        logger.error(f"Anomaly synthesis failed: {e}")
        return _build_list_response(results, question)


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
        return invoke_with_fallback(
            lambda llm: prompt | llm,
            {
                "question": question,
                "history": history or "No history"
            }
        )
    except Exception as e:
        logger.error(f"General synthesis failed: {e}")
        return "I can help you analyze your documents. Try asking about invoices, contracts, or vendors."


def synthesize_multi(
    original_question: str,
    sub_results: list,
    history: str
) -> str:
    """
    Combine multiple sub-question results into one answer.
    Uses Python formatting only — no LLM to prevent hallucination.
    Only uses data actually returned from the database.
    """
    if not sub_results:
        return "No results found."

    sections = []
    for sub in sub_results:
        question = sub.get("question", "")
        results = sub.get("results", [])
        count = sub.get("count", 0)

        if not results:
            sections.append(f"**{question}**\nNo results found.")
            continue

        # Format each sub-result using existing list formatter
        answer = _build_list_response(results, question or "")
        sections.append(f"**{question}**\n{answer}")

    return "\n\n".join(sections)