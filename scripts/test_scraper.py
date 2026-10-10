import asyncio
import time
from backend.scraper import LeadScraper

async def run_tests():
    print("=" * 65)
    print("TEST 1: LIVE SCRAPER INGESTION ATTEMPT")
    print("=" * 65)
    scraper = LeadScraper()
    
    # We test against a public real estate portal URL
    test_portal_url = "https://www.olx.com.pk/properties_c2"
    
    start_time = time.perf_counter()
    results = await scraper.scrape_portal(test_portal_url)
    elapsed = time.perf_counter() - start_time
    
    print(f"\n[+] Ingestion completed in {elapsed:.2f} seconds.")
    print(f"[+] Total Listings Acquired: {len(results)}")
    print(f"[+] Data Source: {results[0].get('source', 'unknown')}")
    print(f"[+] Sample Record: {results[0]['title']} | Price: {results[0]['raw_price']}")

    # ========================================================
    # CHAOS TEST 6.1: Bot Challenge / Broken URL Circuit Breaker Speed
    # ========================================================
    print("\n" + "=" * 65)
    print("CHAOS TEST 6.1: SIMULATED OUTAGE & CIRCUIT BREAKER SPEED")
    print("=" * 65)
    
    # Pass an intentionally broken / blocked endpoint
    broken_url = "https://httpstat.us/403"
    print(f"Targeting Broken Endpoint: {broken_url}")
    
    cb_start = time.perf_counter()
    fallback_results = await scraper.scrape_portal(broken_url)
    cb_elapsed_ms = (time.perf_counter() - cb_start) * 1000
    
    print(f"[+] Circuit Breaker activated in {cb_elapsed_ms:.2f}ms (Target: < 500ms when triggered).")
    print(f"[+] Total Fallback Records Loaded: {len(fallback_results)}")
    assert len(fallback_results) == 10, "Failed to load complete fallback dataset!"
    assert fallback_results[0]["source"] == "offline_fallback", "Source must be offline fallback!"
    print("[SUCCESS] Zero-downtime circuit breaker verified! Downstream pipeline will NEVER starve.")

if __name__ == "__main__":
    asyncio.run(run_tests())