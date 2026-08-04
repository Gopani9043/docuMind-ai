# DocParse Chatbot — Regression Test Checklist

## What this file is for

Every time you change `orchestrator.py`, `query_generator.py`,
`special_intent_classifier.py`, `ranking_selector.py`, `context_resolver.py`,
`query_rewriter.py`, or `response_synthesizer.py` — **fixing one question can
silently break another.** This happened multiple times during development:
adding the `most_important_vendor` special intent broke "which vendor did I
pay the most," and adding ranking-position detection broke "show invoices
above 10000 EUR."

This file is your fixed checklist so you don't have to remember 90 questions
from memory or guess what might have broken. Re-run the flagged subset after
any code change, instead of testing everything every time.

## How to use this file

1. **After any code change** to the files listed above, open this file.
2. Look at the **"Risk tier"** column for each question.
3. Test at minimum every question marked 🔴 or 🟠 that touches the area you
   just changed (see "Depends on" column).
4. Ask Claude (or whoever's debugging) to check the relevant section —
   paste this file's status plus your terminal logs for any question that
   fails.
5. Update the **Status** column (✅ Pass / ❌ Fail / ⚠️ Not tested) and the
   **Last verified** date as you go, so the file stays a living record
   instead of going stale.
6. Commit this file to your repo (`tests/regression_questions.md`) so it
   travels with the code and survives across conversations/sessions.

## Risk tier legend

- 🟢 **Safe** — handled by Step 3a–3h hardcoded keyword handlers, which run
  *before* the special-intent classifier. Changes to classifier/ranking
  logic cannot affect these.
- 🟡 **Medium** — goes through `generate_sql()` (LLM-generated SQL). Could
  be affected by prompt changes in `query_generator.py`, but not by
  classifier/ranking changes.
- 🟠 **High** — passes through `special_intent_classifier.py`. Any new
  category or changed examples in that prompt can silently re-route these.
- 🔴 **Critical** — directly depends on `ranking_selector.py` and/or
  currency conversion (`currency_converter.py`). Most fragile area; most
  past regressions happened here.

---

## LEVEL 1 — Basic queries

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 1 | Show me all invoices | 🟢 | Step 3 fallback / generate_sql | ✅ | |
| 2 | List all contracts | 🟢 | generate_sql | ✅ | |
| 3 | Show all receipts | 🟢 | generate_sql | ✅ | |
| 4 | How many documents do I have? | 🟢 | generate_sql | ✅ | |
| 5 | Show failed documents | 🟢 | generate_sql | ✅ | |
| 6 | What document types do I have? | 🟢 | generate_sql (GROUP BY rule) | ✅ | |
| 7 | Show processing status summary | 🟢 | generate_sql | ✅ | |

## LEVEL 2 — Filtering

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 8 | Show invoices above 10000 EUR | 🔴 | ranking_selector threshold-exclusion rule | ⚠️ RETEST | |
| 9 | Show invoices below 5000 USD | 🔴 | ranking_selector threshold-exclusion rule | ⚠️ RETEST | |
| 10 | Show EUR invoices only | 🟡 | generate_sql | ✅ | |
| 11 | Show invoices from BrightPath Analytics | 🟡 | vendor_matcher, generate_sql | ✅ | |
| 12 | Show contracts from Nordic Systems | 🟡 | vendor_matcher, generate_sql | ✅ | |
| 13 | Show invoices uploaded this month | 🟡 | DATE RULES (d.created_at — intentional) | ✅ | |
| 14 | Show documents uploaded last 30 days | 🟡 | DATE RULES (d.created_at) | ✅ | |
| 15 | Show overdue invoices | 🟢 | Step 3f hardcoded handler | ✅ | |
| 16 | Show contracts expiring soon | 🟢 | Step 3f hardcoded handler | ✅ | |

## LEVEL 3 — Aggregation

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 17 | Which vendor did I pay the most? | 🔴 | special_intent_classifier vs most_important_vendor collision | ⚠️ RETEST | |
| 18 | What is the total amount of all invoices? | 🟢 | Step 3g hardcoded handler | ✅ | |
| 19 | What is the average invoice amount? | 🟡 | generate_sql | ✅ | |
| 20 | Which currency has the highest total? | 🟢 | Step 3h hardcoded handler | ✅ | |
| 21 | How many invoices per vendor? | 🟡 | generate_sql | ✅ | |
| 22 | Show top 5 vendors by total paid | 🔴 | special_intent_classifier vs most_important_vendor collision | ⚠️ RETEST | |
| 23 | What is the total EUR spending? | 🟢 | Step 3g hardcoded handler | ✅ | |
| 24 | Show invoice count by document type | 🟡 | generate_sql (DOCUMENT TYPE RULE) | ✅ | |
| 25 | Which month had the highest invoice total? | 🔴 | spending_trend, ranking_selector metric field | ⚠️ RETEST | |

## LEVEL 4 — Ranking

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 26 | Show the largest invoice | 🟡 | generate_sql (LIMIT 1 rule) | ✅ | |
| 27 | Show the smallest invoice | 🟡 | generate_sql | ✅ | |
| 28 | Show the 10 most expensive invoices | 🟡 | generate_sql, fix_distinct_order_conflict | ✅ | |
| 29 | Show the 5 cheapest invoices | 🟡 | generate_sql, query_rewriter fresh-query guard | ✅ | |
| 30 | What is the second largest invoice? | 🟡 | generate_sql OFFSET rule, query_rewriter ordinal guard | ✅ | |
| 31 | Show the newest invoice | 🟡 | DATE RULES (issue_date, NEWEST/OLDEST rule) | ✅ | |
| 32 | Show the oldest contract | 🟡 | DATE RULES (start_date) | ✅ | |
| 33 | Which vendor appears most often? | 🟡 | generate_sql | ✅ | |
| 34 | Show top 3 vendors in EUR | 🔴 | special_intent_classifier vs most_important_vendor collision | ⚠️ RETEST | |

## LEVEL 6 — Comparative

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 38 | Compare this month vs last month invoices | 🟡 | DATE RULES, response_synthesizer TYPE COMPARISON | ✅ | |
| 39 | Compare EUR vs USD invoice totals | 🟡 | query_rewriter "compare" guard, generate_sql | ✅ | |
| 40 | Compare BrightPath vs FinEdge total spending | 🟡 | currency_keywords/total_keywords exclusion | ✅ | |
| 41 | Which vendor grew most between months? | 🟠 | special_intent_classifier (growth ≠ trend ≠ importance) | ⚠️ RETEST | |
| 42 | Show invoice trend over last 6 months | 🟡 | is_complex_question exclusion (trend patterns) | ✅ | |
| 43 | Compare average invoice amount by currency | 🟡 | currency_keywords narrow match | ✅ | |
| 44 | Which invoices look suspicious? | 🟡 | anomaly_words detection | ✅ | |
| 45 | Find duplicate invoices | 🟢 | Step 3e hardcoded handler | ✅ | |
| 46 | Which invoice repeats most? | 🟢 | Step 3e hardcoded handler (is_most) | ✅ | |
| 47 | Are there any unusual amounts? | 🟡 | anomaly_words, detect_outliers | ✅ | |
| 48 | Find vendors appearing under slightly different spellings | 🟢 | Step 3d hardcoded handler | ✅ | |
| 49 | Which invoices are missing due dates? | 🟡 | generate_sql | ✅ | |
| 50 | Which contracts are missing values? | 🟡 | generate_sql | ✅ | |
| 51 | Show invoices with missing vendor names | 🟡 | generate_sql | ✅ | |

## LEVEL 8 — Cross-document intelligence

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 52 | Does BrightPath Analytics appear in contracts too? | 🟡 | fix_cross_document_vendor_match | ✅ | |
| 53 | Which vendors have both invoices and contracts? | 🟡 | generate_sql EXISTS pattern | ✅ | |
| 54 | Show contracts from vendors who also sent invoices above 50000 | 🟡 | fix_cross_document_exists (amount+EXISTS) | ✅ | |
| 55 | Find vendors in contracts but not in invoices | 🟡 | fix_cross_document_exists (NOT EXISTS detection) | ✅ | |
| 56 | Show all documents from EuroData AG | 🟡 | vendor_matcher (watch for false positives like "euro") | ✅ | |
| 57 | Which vendor has the most total documents? | 🟡 | generate_sql | ✅ | |

## LEVEL 9 — Complex multi-part

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 58 | Find the vendor with highest spending this year, show their latest invoice, summarize their contracts | 🟡 | is_complex_question, decompose_question, vendor entity injection | ✅ | |
| 59 | Show all EUR invoices above 10000, find duplicates, show the oldest one | 🟡 | decompose_question | ✅ | |
| 60 | Compare top 3 vendors by EUR spending and show their contract values | 🟡 | Step 3j cross_doc_ranking | ✅ | |
| 61 | Find overdue invoices from vendors who also have expiring contracts | 🟡 | has_overdue_extra / has_expiring_extra exclusions | ✅ | |
| 62 | Show monthly invoice trend for BrightPath Analytics | 🟡 | is_complex_question trend exclusion, MONTHLY TREND RULE | ✅ | |

## LEVEL 10 — Vague and conversational

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 63 | How much money did we spend? | 🟢 | Step 3g total_keywords | ✅ | |
| 64 | Any expensive contracts? | 🟡 | document_type_mismatch standalone forcing | ✅ | |
| 65 | What are my biggest bills? | 🔴 | special_intent_classifier → biggest_bills, currency_converter | ✅ | |
| 66 | Do we have contracts in euros? | 🟡 | vendor_matcher currency-word stoplist | ✅ | |
| 67 | Which suppliers appear often? | 🟡 | generate_sql | ✅ | |
| 68 | What files failed? | 🟢 | generate_sql, no-results fallback | ✅ | |
| 69 | Show me something interesting about our invoices | 🔴 | special_intent_classifier → something_interesting | ✅ | |
| 70 | Any missing data in our documents? | 🔴 | special_intent_classifier → missing_data | ✅ | |
| 71 | Which vendors should I be worried about? | 🔴 | special_intent_classifier → vendor_risk, ranking_selector | ✅ | |
| 72 | Is our spending increasing or decreasing? | 🔴 | special_intent_classifier → spending_trend, currency_converter | ✅ | |
| 73 | Who is our most important vendor? | 🔴 | special_intent_classifier → most_important_vendor, ranking_selector | ✅ | |
| 74 | Are there any red flags in our documents? | 🟠 | could overlap vendor_risk / missing_data / something_interesting | ✅ | |
| 75 | Summarize our financial situation | 🟠 | could overlap something_interesting | ✅ | |

## LEVEL 11 — Stress test with synonyms

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 76 | Show me all rechnungen | 🟡 | SYNONYMS rule in generate_sql | ✅ | |
| 77 | List all agreements | 🟡 | SYNONYMS rule | ✅ | |
| 78 | Show bills from last month | 🟡 | DATE RULES (issue_date), generate_sql | ✅ | |
| 79 | Which lieferant did we pay most? | 🟠 | "lieferant" NOT in SYNONYMS list — likely fails | ❌ KNOWN BUG | |
| 80 | Show kassenbon from Bayern Software | 🟡 | SYNONYMS rule (kassenbon=receipt) | ⚠️ untested | |
| 81 | Find all fakturas above 1000 | 🟡🔴 | SYNONYMS + threshold-exclusion overlap | ✅ | |
| 82 | Which vertrag expires soon? | 🟢 | Step 3f expiring_keywords — "expires" not in list | ❌ KNOWN BUG | |

## LEVEL 12 — Edge cases

| # | Question | Risk | Depends on | Status | Last verified |
|---|---|---|---|---|---|
| 83 | Show invoices (no filter) | 🟢 | generate_sql | ✅ | |
| 84 | Show (one word) | 🟢 | synthesize_error fallback | ✅ | |
| 85 | hello | 🟢 | classify_intent → greeting | ✅ | |
| 86 | What can you do? | 🟢 | classify_intent → general | ✅ | |
| 87 | Show invoices from 2099 | 🟢 | generate_sql, no-results fallback | ✅ | |
| 88 | Show invoices above 999999999 | 🔴 | same threshold-exclusion logic as Q8/Q9 | ⚠️ RETEST | |
| 89 | Compare EUR vs EUR | 🟡 | response_synthesizer TYPE COMPARISON edge case | ✅ | |
| 90 | Show the 100th invoice | 🟡 | ranking_selector single_rank, OFFSET handling | ✅ | |

---

## Known confirmed bugs (not yet fixed)

- **Q79** — "lieferant" (German for supplier/vendor) is not in the
  `SYNONYMS` block in `query_generator.py`. Add it next to `vendor =
  supplier|seller|company`.
- **Q82** — `expiring_keywords` in `orchestrator.py` Step 3f only contains
  `["expiring", "expiring soon", "ending soon", "renewal due"]` — missing
  the bare word `"expires"`. Add it to the list.

## Untested

- **Q80** — never confirmed with real logs.
- **Level 5 (follow-up chain questions, originally #35–37)** — entirely
  missing from this checklist. These test multi-turn context memory
  (vendor carried across turns, filters preserved across follow-ups) and
  were historically the most fragile area. Recommend adding a Level 5
  section back once Level 1–12 are all green.

## Currently flagged for retest (🔴/🟠 not yet re-verified after recent changes)

Q8, Q9, Q17, Q22, Q25, Q34, Q41, Q88

Test these first. For each one, paste:
1. The exact question you typed
2. The chatbot's full response
3. The terminal log block for that request (from `Processing:` to the
   final `COMMIT`)

That's enough for a precise diagnosis without re-explaining the whole
system each time.
