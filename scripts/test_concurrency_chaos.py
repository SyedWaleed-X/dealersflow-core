import asyncio
import time
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from backend.main import app
from backend.core.database import AsyncSessionLocal, engine
from backend.core.config import Settings, ComputeProfile


# ==========================================
# 1. DATABASE RECORD VERIFICATION (TASK 14.3)
# ==========================================
async def verify_database_record_integrity():
    print("=" * 80)
    print("TASK 14.3: POSTGRESQL RECORD INTEGRITY & VECTOR DIMENSION VERIFICATION")
    print("=" * 80)

    async with AsyncSessionLocal() as session:
        # Fetch the most recent property record
        query = text("""
            SELECT id, city, society, price_pkr, area_sqft, 
                   vector_dims(embedding) AS dims, 
                   features->>'verified_source' AS source,
                   features->>'data_source' AS data_source
            FROM properties
            WHERE embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1;
        """)
        res = await session.execute(query)
        record = res.mappings().first()

        assert record is not None, "No valid embedded records found in database!"

        print(f"[+] Latest Record ID    : {record['id']}")
        print(f"[+] City & Society      : {record['city']} | {record['society']}")
        print(f"[+] Standard Price      : {record['price_pkr']:,} PKR")
        print(f"[+] Standard Area       : {record['area_sqft']} sqft")
        print(f"[+] Vector Dimensions   : {record['dims']} (Target: 768)")

        assert record["dims"] == 768, f"Dimensionality mismatch! Expected 768, got {record['dims']}"
        assert record["price_pkr"] > 0, "Price must be positive!"
        print("[SUCCESS] Database record verified: 768-dim EmbeddingGemma 2 vector persisted correctly!")


# ==========================================
# 2. CONCURRENCY STRESS TEST (TASK 14.4)
# ==========================================
async def run_concurrency_stress_test():
    print("\n" + "=" * 80)
    print("TASK 14.4: CONCURRENCY STRESS TEST (5 Simultaneous Voice Leads)")
    print("=" * 80)

    prompts = [
        "Aoa bhai, DHA Phase 6 mein 1 kanal plot bechna hai demand 8.5 crore tariq 03001234567",
        "Mujhe Bahria Town Phase 7 mein 10 marla makan chahiye budget 2.8 crore Farhan 03219876543",
        "Johar Town Block B mein 5 marla plot for sale demand 95 lakh cash ready",
        "Islamabad Sector F-7 mein 2 kanal luxury villa chahiye budget 24 crore Malik Sb",
        "Gujranwala Citi Housing mein 1 kanal commercial plot bechna hai demand 3.5 crore"
    ]

    transport = ASGITransport(app=app)

    async def send_lead(client: AsyncClient, text_prompt: str, idx: int):
        t0 = time.perf_counter()
        res = await client.post("/api/leads/process-voice", data={"raw_text": text_prompt})
        t_el = (time.perf_counter() - t0) * 1000
        return idx, res.status_code, t_el, res.json() if res.status_code == 201 else res.text

    print(f"[*] Dispatching 5 concurrent requests across worker threadpool...")
    start_total = time.perf_counter()

    async with AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
        tasks = [send_lead(client, p, i) for i, p in enumerate(prompts, 1)]
        results = await asyncio.gather(*tasks)

    total_elapsed = time.perf_counter() - start_total
    print(f"\n[+] All 5 concurrent requests completed in {total_elapsed:.2f} seconds!")

    for idx, status_code, elapsed_ms, body in results:
        status_label = "PASS (201 Created)" if status_code == 201 else f"FAIL ({status_code})"
        print(f"  • Request #{idx}: Status {status_label} | Latency: {elapsed_ms:.1f} ms")
        assert status_code == 201, f"Concurrent request #{idx} failed: {body}"

    print("[SUCCESS] Zero connection pool deadlocks! All 5 concurrent transactions committed safely.")


# ==========================================
# 3. CHAOS TEST 14.2: ENVIRONMENT FALLBACK
# ==========================================
def test_missing_env_fallback():
    print("\n" + "=" * 80)
    print("CHAOS TEST 14.2: MISSING ENVIRONMENTAL PROFILE FALLBACK")
    print("=" * 80)

    # Instantiate Settings with an empty dictionary (simulating a completely missing .env file)
    fallback_settings = Settings(_env_file=None)

    print(f"[*] Default Profile Resolved: {fallback_settings.COMPUTE_PROFILE}")
    print(f"[*] Default Model Resolved  : {fallback_settings.active_llm_model}")
    print(f"[*] Default Whisper Size    : {fallback_settings.whisper_model_size}")

    assert fallback_settings.COMPUTE_PROFILE == ComputeProfile.BEAST
    assert fallback_settings.whisper_model_size == "large-v3-turbo"
    print("[SUCCESS] Settings factory safely defaults without crashing or throwing KeyError!")


async def main():
    await verify_database_record_integrity()
    await run_concurrency_stress_test()
    test_missing_env_fallback()

    print("\n" + "=" * 80)
    print("[DAY 14 COMPLETE] Systemic Integration, Concurrency & Chaos Tests 100% Green!")
    print("=" * 80)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())