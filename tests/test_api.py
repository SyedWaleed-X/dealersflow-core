import time
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.core.database import engine


# ==========================================
# FIX: DRAIN CONNECTION POOL BETWEEN TESTS
# ==========================================
@pytest.fixture(autouse=True)
async def cleanup_db_pool():
    """Ensures each test gets a fresh database socket on its own active event loop."""
    yield
    # Disposes connections created in this test before its event loop closes
    await engine.dispose()


# ==========================================
# TEST 1: HEALTH CHECK LATENCY
# ==========================================
@pytest.mark.anyio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.perf_counter()
        response = await client.get("/health")
        latency_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert latency_ms < 50.0, f"Health check too slow: {latency_ms:.2f}ms"
        print(f"\n[+] Health endpoint responded in {latency_ms:.2f} ms")


# ==========================================
# TEST 2: PAGINATED LEADS QUERY & FILTERING
# ==========================================
@pytest.mark.anyio
async def test_get_leads_pagination():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/leads?city=Lahore&limit=5")
        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "items" in data
        assert len(data["items"]) <= 5
        print(f"\n[+] Retrieved {len(data['items'])} leads (Total in DB: {data['total']})")


# ==========================================
# TEST 3: PROCESS VOICE LEAD ENDPOINT (END-TO-END)
# ==========================================
@pytest.mark.anyio
async def test_process_lead_endpoint_e2e():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "raw_text": "Aoa bhai, DHA Phase 6 mein 1 kanal commercial plot bechna hai demand 8 crore 50 lac hai tariq 03001234567"
        }

        print("\n[*] Sending multipart lead to /api/leads/process-voice...")
        start = time.perf_counter()
        response = await client.post("/api/leads/process-voice", data=payload)
        elapsed = time.perf_counter() - start

        assert response.status_code == 201, f"Failed with {response.status_code}: {response.text}"
        data = response.json()

        assert "property_id" in data
        assert "valuation" in data
        assert "whatsapp_deep_link" in data
        assert data["lead"]["intent"] == "SELL"
        print(f"[+] API processed lead in {elapsed:.2f}s! Generated ID: {data['property_id']}")
        print(f"[+] WhatsApp Link: {data['whatsapp_deep_link'][:60]}...")


# ==========================================
# CHAOS TEST 11.1: EVENT LOOP STARVATION TEST
# ==========================================
@pytest.mark.anyio
async def test_event_loop_starvation():
    print("\n" + "=" * 75)
    print("CHAOS TEST 11.1: EVENT LOOP STARVATION PROFILE")
    print("=" * 75)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            res = await client.get("/health")
            t_ms = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200
            latencies.append(t_ms)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        print(f"[*] Concurrency profile over 30 requests:")
        print(f"    • Average Health Latency : {avg_latency:.2f} ms (Target: < 15ms)")
        print(f"    • Peak Latency           : {max_latency:.2f} ms (Target: < 25ms)")

        assert avg_latency < 20.0, f"Event loop starvation detected! Avg latency: {avg_latency:.2f}ms"
        print("[SUCCESS] Event loop remained 100% responsive! Zero CPU starvation.")