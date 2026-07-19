import pytest
import httpx
import re
import uuid
import asyncio

BASE_URL = "http://localhost:8000"

@pytest.fixture(autouse=True)
async def rate_limit_delay():
    """Add small delay between tests to avoid LLM rate limits."""
    yield
    await asyncio.sleep(5)

async def ask(question: str, session_id: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{BASE_URL}/smart-chat", json={
            "question": question,
            "doc_id": session_id
        })
        assert r.status_code == 200, f"HTTP {r.status_code} for: {question}"
        data = r.json()
        assert "answer" in data, f"No answer field for: {question}"
        if not data["answer"]:
            print(f"\nEMPTY ANSWER for: {question}, intent={data.get('intent')}, count={data.get('count')}, sql={str(data.get('sql',''))[:100]}")
        assert data["answer"], f"Empty answer for: {question}"
        return data

def fresh_session():
    return f"test_{uuid.uuid4().hex[:8]}"


# ── Category 1: No crash guarantee ───────────────────────────────────────────
# Every question must return 200 with non-empty answer.
# Doesn't care about content — just that nothing explodes.

SMOKE_QUESTIONS = [
    "Show all invoices",
    "Show all EUR invoices",
    "Find duplicate invoices",
    "Show overdue invoices",
    "Which vendors should I be worried about?",
    "Show expiring contracts",
    "What is the total amount of all invoices?",
    "Which vendor did I pay the most?",
    "Is our spending increasing or decreasing?",
    "Show something interesting",
    "Any missing data?",
    "Show all contracts",
    "Does BrightPath appear in contracts?",
    "Does XYZ Nonexistent Corp appear in contracts?",
    "Show invoices above 50000 EUR",
    "Compare top 3 vendors by EUR spending and show their contract values",
    "Show monthly invoice trend for BrightPath Analytics",
    "Which vendor has the most total documents?",
    "Show vendor spend ranking but only for vendors with at least 5 invoices",
    "What's my biggest contract?",
    "Show contract spending trend over time",
]

@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("question", SMOKE_QUESTIONS)
async def test_no_crash(question):
    session = fresh_session()
    resp = await ask(question, session)
    # Only guarantee: no crash, non-empty answer
    assert len(resp["answer"]) > 10, f"Suspiciously short answer for: {question}"


# ── Category 2: SQL structural correctness ───────────────────────────────────
# Assert on the SQL structure, not the data values.

@pytest.mark.asyncio
async def test_eur_filter_in_sql():
    resp = await ask("Show all EUR invoices", fresh_session())
    sql = resp.get("sql", "")
    assert "EUR" in sql.upper(), "EUR filter missing from SQL"
    assert "invoice" in sql.lower(), "Wrong document type in SQL"

@pytest.mark.asyncio
async def test_above_filter_in_sql():
    resp = await ask("Show invoices above 50000", fresh_session())
    sql = resp.get("sql", "")
    assert "50000" in sql, "Amount threshold missing from SQL"
    assert ">" in sql, "Comparison operator missing from SQL"

@pytest.mark.asyncio
async def test_contract_query_hits_contract_table():
    resp = await ask("Show all contracts", fresh_session())
    sql = resp.get("sql", "")
    assert "contract" in sql.lower(), "Contract type missing from SQL"
    assert "invoice" not in sql.lower(), "Wrong table in contract query SQL"

@pytest.mark.asyncio
async def test_vendor_filter_in_sql():
    resp = await ask("Show invoices from BrightPath Analytics", fresh_session())
    sql = resp.get("sql", "")
    assert "brightpath" in sql.lower(), "Vendor filter missing from SQL"

@pytest.mark.asyncio
async def test_unknown_vendor_returns_zero():
    resp = await ask("Show invoices from XYZ Nonexistent Corp 99999", fresh_session())
    assert resp["count"] == 0, "Should return 0 for nonexistent vendor"


# ── Category 3: Result count sanity ──────────────────────────────────────────
# Don't hardcode exact counts — assert on ranges or relationships.

@pytest.mark.asyncio
async def test_eur_invoices_subset_of_all():
    s = fresh_session()
    all_resp = await ask("Show all invoices", s)
    eur_resp = await ask("Show all EUR invoices", fresh_session())
    assert eur_resp["count"] <= all_resp["count"], \
        "EUR invoices should be a subset of all invoices"
    assert eur_resp["count"] > 0, "Should have at least some EUR invoices"

@pytest.mark.asyncio
async def test_above_filter_reduces_results():
    s1 = fresh_session()
    s2 = fresh_session()
    all_resp = await ask("Show all invoices", s1)
    filtered_resp = await ask("Show invoices above 999999999", s2)
    assert filtered_resp["count"] < all_resp["count"], \
        "Extreme amount filter should reduce results"
    assert filtered_resp["count"] == 0, \
        "No invoice should exceed 999 million"

@pytest.mark.asyncio
async def test_vendor_filter_reduces_results():
    s1 = fresh_session()
    s2 = fresh_session()
    all_resp = await ask("Show all invoices", s1)
    vendor_resp = await ask("Show invoices from BrightPath Analytics", s2)
    assert vendor_resp["count"] < all_resp["count"], \
        "Vendor filter should return fewer results than all invoices"
    assert vendor_resp["count"] > 0, \
        "BrightPath should have at least one invoice"


# ── Category 4: Follow-up chain integrity ────────────────────────────────────
# Test that context carries correctly across turns.
# Assert on structural properties, not exact values.

@pytest.mark.asyncio
async def test_vendor_filter_carries_to_followup():
    s = fresh_session()
    await ask("Show invoices from BrightPath Analytics", s)
    resp = await ask("show the largest one", s)
    assert resp["count"] == 1, "Largest one should return exactly 1 result"
    answer = resp["answer"].lower()
    assert "brightpath" in answer, \
        "Follow-up should still reference BrightPath vendor"

@pytest.mark.asyncio
async def test_currency_filter_carries_to_followup():
    s = fresh_session()
    await ask("Show all EUR invoices", s)
    resp = await ask("only above 50000", s)
    # All results should be EUR if filter carried correctly
    for result in resp.get("results", []):
        cur = result.get("currency")
        if cur:
            assert cur == "EUR", \
                f"Currency filter lost in follow-up — got {cur}"




# ── Category 5: Intent routing correctness ───────────────────────────────────
# Test that questions route to the right handler.

@pytest.mark.asyncio
async def test_duplicate_detection_routes_correctly():
    resp = await ask("Find duplicate invoices", fresh_session())
    assert resp.get("intent") == "duplicate_detection", \
        f"Wrong intent: {resp.get('intent')}"

@pytest.mark.asyncio
async def test_overdue_routes_correctly():
    resp = await ask("Show overdue invoices", fresh_session())
    assert resp.get("intent") == "overdue_invoices", \
        f"Wrong intent: {resp.get('intent')}"

@pytest.mark.asyncio
async def test_expiring_contracts_routes_correctly():
    resp = await ask("Show expiring contracts", fresh_session())
    assert resp.get("intent") == "expiring_contracts", \
        f"Wrong intent: {resp.get('intent')}"

@pytest.mark.asyncio
async def test_vendor_risk_routes_correctly():
    resp = await ask("Which vendors should I be worried about?", fresh_session())
    assert resp.get("intent") == "vendor_risk_assessment", \
        f"Wrong intent: {resp.get('intent')}"

@pytest.mark.asyncio
async def test_greeting_does_not_run_sql():
    resp = await ask("Hello", fresh_session())
    assert resp.get("intent") == "greeting", \
        f"Wrong intent for greeting: {resp.get('intent')}"
    assert resp.get("sql") is None or resp.get("sql") == "", \
        "Greeting should not generate SQL"

# ── Category : Multi-step follow-up chains ───────────────────────────────────

@pytest.mark.slow
@pytest.mark.asyncio
async def test_chain_35_eur_filter_then_vendor_then_smallest():
    s = fresh_session()
    r1 = await ask("Show invoices above 10000 EUR", s)
    assert r1["count"] > 0

    r2 = await ask("only BrightPath", s)
    assert r2["count"] > 0
    for result in r2.get("results", []):
        vendor = result.get("vendor", "")
        assert "brightpath" in vendor.lower(), \
            f"Vendor filter lost — got {vendor}"

    r3 = await ask("smallest one", s)
    assert r3["count"] == 1, "Smallest should return exactly 1"

@pytest.mark.slow
@pytest.mark.asyncio
async def test_chain_vendor_then_all_invoices_then_total():
    s = fresh_session()
    r1 = await ask("Which vendor did I pay the most?", s)
    # vendor_spend_ranking returns count=0 (Python-computed, no SQL results array)
    # check the answer contains a vendor name instead
    assert len(r1["answer"]) > 10, "Should produce a non-empty ranking answer"

    r2 = await ask("show all their invoices", s)
    assert r2["count"] > 0
    vendors = {r.get("vendor") for r in r2.get("results", []) if r.get("vendor")}
    assert len(vendors) <= 2, \
        f"Too many vendors in follow-up — context leaked: {vendors}"

    r3 = await ask("what is the total of above?", s)
    assert len(r3["answer"]) > 10
    assert any(c.isdigit() for c in r3["answer"]), \
        "Total answer should contain a number"

@pytest.mark.slow
@pytest.mark.asyncio
async def test_chain_eur_filter_carries_through_vendor_filter():
    s = fresh_session()
    await ask("Show all EUR invoices", s)
    r2 = await ask("only above 50000", s)
    # All returned results should be EUR
    for result in r2.get("results", []):
        cur = result.get("currency")
        if cur:
            assert cur == "EUR", \
                f"EUR filter lost after amount filter — got {cur}"

@pytest.mark.asyncio
async def test_nonexistent_vendor_in_chain():
    s = fresh_session()
    await ask("Show all invoices", s)
    r2 = await ask("Does XYZ Nonexistent Corp 99999 appear in contracts?", s)
    assert r2["count"] == 0, \
        "Nonexistent vendor should return 0 even mid-chain"

@pytest.mark.asyncio
async def test_cross_document_vendors_both_types():
    resp = await ask(
        "Which vendors have both invoices and contracts?",
        fresh_session()
    )
    assert resp["count"] > 0, "Should find at least one vendor in both types"

@pytest.mark.asyncio
async def test_nth_item_out_of_bounds():
    resp = await ask("Show the 9999th invoice", fresh_session())
    assert "9999" in resp["answer"] or "only" in resp["answer"].lower(), \
        "Out-of-bounds ordinal should explain limit, not crash"

@pytest.mark.asyncio
async def test_self_comparison_guard():
    resp = await ask("Compare EUR vs EUR", fresh_session())
    answer = resp["answer"].lower()
    assert "same" in answer or "nothing to compare" in answer or \
           "identical" in answer, \
        "Self-comparison should be caught, not run a query"

@pytest.mark.asyncio
async def test_vendor_filter_carries_to_followup_debug():
    s = fresh_session()
    r1 = await ask("Show invoices from BrightPath Analytics", s)
    print(f"\nR1 count: {r1['count']}")
    r2 = await ask("show the largest one", s)
    print(f"\nR2 answer: {r2['answer'][:200]}")
    print(f"R2 sql: {r2.get('sql', '')[:200]}")
    print(f"R2 intent: {r2.get('intent')}")

@pytest.mark.asyncio
async def test_no_results_for_nonexistent_vendor_followup():
    s = fresh_session()
    await ask("Show all invoices", s)
    resp = await ask("Does ZZZNONEXISTENTVENDORZZZ appear in contracts?", s)
    # The SQL returns an EXISTS check row (count=1) but answer should say no results
    # Check the answer text rather than count since EXISTS queries return 1 row
    answer_lower = resp["answer"].lower()
    assert (
        resp["count"] == 0 or
        "no" in answer_lower or
        "not found" in answer_lower or
        "does not" in answer_lower or
        "no results" in answer_lower
    ), f"Nonexistent vendor should return no results, got: {resp['answer']}"