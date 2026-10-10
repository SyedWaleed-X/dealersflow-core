import asyncio
import time
from sqlalchemy import text
from backend.core.database import AsyncSessionLocal, engine
from backend.ai.matcher import LeadMatcher
from backend.ai.embedder import embedding_engine

async def run_tests():
    print("=" * 75)
    print("TEST 1: TWO-STAGE HYBRID LEAD MATCHING VERIFICATION")
    print("=" * 75)

    # Simulated Incoming Buyer Lead
    test_intent = "BUY"
    test_city = "Lahore"
    test_budget = 25_000_000  # 2.5 Crore PKR
    test_preference = "corner plot near mosque facing park"

    print(f"[*] Incoming Inquiry: Intent={test_intent} | City={test_city} | Budget={test_budget:,} PKR")
    print(f"[*] Qualitative Preferences: '{test_preference}'\n")

    start_time = time.perf_counter()
    matches = await LeadMatcher.match_lead(
        intent=test_intent,
        city=test_city,
        target_budget_pkr=test_budget,
        tolerance_percentage=0.30,  # +/- 30% price flexibility
        preference_text=test_preference,
        limit=5
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"[+] Hybrid Match completed in {elapsed_ms:.2f} ms!")
    print(f"[+] Total Candidates Matched: {len(matches)}\n")

    assert len(matches) > 0, "Expected at least 1 match from database inventory!"

    for i, m in enumerate(matches, 1):
        print(f"  Rank #{i}: {m['city'].upper()} | {m['society']} | {m['price_pkr']:,} PKR | {m['prop_type']}")
        print(f"          Similarity: {m['similarity_score']} | Cosine Distance: {m['cosine_distance']:.4f}")
        if m.get("listing_url"):
            print(f"          Link: {m['listing_url']}")

        # Strict relational boundary assertions:
        assert m["city"].lower() == test_city.lower(), f"CROSS-CITY LEAKAGE! Found {m['city']} in {test_city} query!"
        assert 17_500_000 <= m["price_pkr"] <= 32_500_000, f"PRICE BOUNDARY BREACH! {m['price_pkr']} outside range!"

    print("\n[SUCCESS] Strict relational boundaries verified: 0% cross-city leakage, 100% budget adherence!")

    # ========================================================
    # TEST 2: EXPLAIN ANALYZE ON HYBRID QUERY
    # ========================================================
    print("\n" + "=" * 75)
    print("TEST 2: EXPLAIN ANALYZE QUERY EXECUTION PLAN")
    print("=" * 75)

    query_vector = str(embedding_engine.embed_text(f"{test_city} {test_preference}"))
    explain_sql = text(f"""
        EXPLAIN ANALYZE
        SELECT id, city, price_pkr, (embedding <=> '{query_vector}'::vector) AS dist
        FROM properties
        WHERE LOWER(city) = 'lahore'
          AND intent = 'SELL'
          AND price_pkr BETWEEN 17500000 AND 32500000
        ORDER BY dist ASC
        LIMIT 5;
    """)

    async with AsyncSessionLocal() as session:
        res = await session.execute(explain_sql)
        for row in res.fetchall():
            print(f"  {row[0]}")

    # ========================================================
    # CHAOS TEST 7.1: Zero Candidate Hard-Filter Recovery
    # ========================================================
    print("\n" + "=" * 75)
    print("CHAOS TEST 7.1: ZERO CANDIDATE EMPTY-SET RECOVERY")
    print("=" * 75)
    # Impossible budget query (10 PKR to 50 PKR plot in Lahore)
    impossible_matches = await LeadMatcher.match_lead(
        intent="BUY",
        city="Lahore",
        target_budget_pkr=20,
        tolerance_percentage=0.10,
        preference_text="unrealistic query",
        limit=5
    )
    print(f"[*] Impossible Budget Query Returned: {len(impossible_matches)} results.")
    assert len(impossible_matches) == 0, "Expected empty result set on impossible constraints!"
    print("[SUCCESS] Zero candidate recovery passed without database exceptions or unhandled errors.")

    # ========================================================
    # CHAOS TEST 7.2: Vector Drift on Empty Strings
    # ========================================================
    print("\n" + "=" * 75)
    print("CHAOS TEST 7.2: VECTOR DRIFT ON EMPTY STRINGS")
    print("=" * 75)
    empty_vec = embedding_engine.embed_text("   ")
    assert len(empty_vec) == 768, "Empty text did not return 768 dimensions!"
    assert not any(x != x for x in empty_vec), "Vector contains NaNs on empty input!"
    print("[SUCCESS] Empty string vector drift prevented: Clean 768-dim baseline vector generated with 0 NaNs.")

    print("\n" + "=" * 75)
    print("[DAY 7 COMPLETE] All Semantic Search and Hybrid Matcher tests 100% green!")
    print("=" * 75)

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_tests())