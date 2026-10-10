import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from backend.main import app
from backend.core.database import engine, AsyncSessionLocal
from backend.domain.schemas import ExtractedPropertyLead, PropertyIntent


@pytest.fixture(autouse=True)
async def cleanup_db_pool():
    yield
    await engine.dispose()


# ==========================================
# 1. PAYLOAD SIZE LIMITER TEST (HTTP 413)
# ==========================================
@pytest.mark.anyio
async def test_payload_size_limit_rejection():
    """
    Simulates an oversized upload exceeding the 25MB security ceiling.
    Verifies that PayloadSizeLimitMiddleware immediately rejects it with HTTP 413.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Mock a 26MB Content-Length header (26 * 1024 * 1024 = 27,262,976 bytes)
        headers = {"content-length": "27262976"}
        response = await client.post("/api/leads/process-voice", headers=headers, content=b"")

        assert response.status_code == 413, f"Expected 413, got {response.status_code}: {response.text}"
        data = response.json()
        assert "Payload too large" in data["detail"]
        print(f"\n[+] Security Middleware Blocked Oversized Upload: {response.status_code} Request Entity Too Large")


# ==========================================
# 2. SQL INJECTION PROBE (CHAOS TEST 15.2)
# ==========================================
@pytest.mark.anyio
async def test_sql_injection_probe():
    """
    Injects classic SQL injection payloads into property metadata.
    Asserts that parameterized queries neutralize the attack and properties table remains intact.
    """
    malicious_society = "DHA'; DROP TABLE properties; --"
    malicious_title = "' OR '1'='1"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "city": "Lahore",
            "society": malicious_society,
            "property_type": "PLOT",
            "intent": "BUY",
            "price_pkr": 25000000,
            "area_sqft": 4500.0,
            "client_name": "Attacker",
            "client_phone": "03001234567",
            "notes": malicious_title
        }

        response = await client.post("/api/leads/manual", json=payload)
        assert response.status_code == 201

    # Verify that the properties table was NOT dropped and still exists!
    async with AsyncSessionLocal() as session:
        check_table = await session.execute(text("SELECT count(*) FROM properties;"))
        count = check_table.scalar()
        assert count > 0, "Security Failure: Table was dropped or emptied!"

        # Verify the malicious string was saved as literal, harmless text
        verify_row = await session.execute(
            text("SELECT society FROM properties WHERE society = :soc LIMIT 1;"),
            {"soc": malicious_society}
        )
        saved_soc = verify_row.scalar()
        assert saved_soc == malicious_society

    print(f"\n[+] SQL Injection Neutralized: Malicious payload safely escaped as literal text string.")
    print(f"[+] Verified properties table intact with {count} active records.")


# ==========================================
# 3. PROMPT INJECTION EVASION TEST (TASK 15.4)
# ==========================================
@pytest.mark.anyio
async def test_prompt_injection_guardrails():
    """
    Feeds an adversarial system prompt evasion attempt into Pydantic validation.
    Verifies that invalid enum values like 'HACKED' are rejected by schema guardrails.
    """
    adversarial_dict = {
        "intent": "HACKED_INTENT",  # Invalid enum value!
        "property_type": "PLOT",
        "city": "Lahore",
        "society": "DHA",
        "raw_price": "0 PKR"
    }

    try:
        ExtractedPropertyLead(**adversarial_dict)
        pytest.fail("Pydantic schema failed to reject invalid adversarial intent!")
    except Exception as e:
        assert "Input should be 'BUY' or 'SELL'" in str(e)
        print(f"\n[+] Prompt Injection Defended: Schema rejected malicious intent: {e}")