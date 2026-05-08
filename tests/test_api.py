import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import pytest
from httpx import AsyncClient, ASGITransport
from main import app

SAMPLES = Path(__file__).parent.parent / "sample_documents"

@pytest.mark.asyncio
async def test_health_check():
    """Health endpoint should return healthy."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_root_endpoint():
    """Root endpoint should return API info."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "DocParse" in r.json()["message"]

@pytest.mark.asyncio
async def test_upload_invalid_type():
    """Uploading a .txt file should return 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/upload",
            files={"file": ("test.txt", b"hello", "text/plain")}
        )
    assert r.status_code == 400

@pytest.mark.asyncio
async def test_results_invalid_id():
    """Fetching results with invalid UUID should return 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/results/not-a-valid-uuid")
    assert r.status_code == 400

@pytest.mark.asyncio
async def test_results_not_found():
    """Fetching results for non-existent doc should return 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/results/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404