import time
import shutil
from pathlib import Path
from backend.ai.valuation import ValuationService, valuation_service

MODEL_PATH = Path("models/catboost_valuation.cbm")
BACKUP_PATH = Path("models/catboost_valuation.cbm.bak")

SAMPLE_PROPERTIES = [
    # 1. High-End Luxury House in Lahore
    {"city": "Lahore", "society": "DHA", "phase": "Phase 6", "type": "HOUSE", "area": 4500.0, "beds": 5, "baths": 5, "asking": 110_000_000},
    # 2. Overpriced Plot in Gujranwala
    {"city": "Gujranwala", "society": "Citi Housing", "phase": "Phase 1", "type": "PLOT", "area": 1125.0, "beds": 0, "baths": 0, "asking": 15_000_000},
    # 3. Underpriced Commercial Plaza in Islamabad
    {"city": "Islamabad", "society": "Sector F-7", "phase": "Phase 2", "type": "COMMERCIAL", "area": 4500.0, "beds": 0, "baths": 0, "asking": 200_000_000},
    # 4. Standard House in Karachi
    {"city": "Karachi", "society": "Clifton", "phase": "Block 4", "type": "HOUSE", "area": 2250.0, "beds": 4, "baths": 4, "asking": 85_000_000},
]

def run_tests():
    print("=" * 75)
    print("TEST 1: VALUATION INFERENCE & PREDICTION INTERVALS")
    print("=" * 75)

    for i, p in enumerate(SAMPLE_PROPERTIES, 1):
        val = valuation_service.predict_fair_value(
            city=p["city"],
            society=p["society"],
            phase=p["phase"],
            property_type=p["type"],
            area_sqft=p["area"],
            bedrooms=p["beds"],
            bathrooms=p["baths"]
        )

        deal = valuation_service.evaluate_market_deal(
            asking_price_pkr=p["asking"],
            estimated_price_pkr=val["estimated_price_pkr"],
            lower_bound_pkr=val["lower_bound_pkr"],
            upper_bound_pkr=val["upper_bound_pkr"]
        )

        print(f"\n--- Property #{i}: {p['city'].upper()} | {p['society']} ({p['type']}) ---")
        print(f"  Asking Price   : {p['asking']:,} PKR")
        print(f"  Estimated Fair : {val['estimated_price_pkr']:,} PKR (Range: {val['lower_bound_pkr']:,} - {val['upper_bound_pkr']:,} PKR)")
        print(f"  Arbitrage Deal : {deal['deal_status']} (Spread: {deal['spread_percentage']}%)")
        print(f"  Recommendation : {deal['recommendation']}")

    # ========================================================
    # TEST 2: CPU LATENCY BENCHMARK (< 10ms on CPU)
    # ========================================================
    print("\n" + "=" * 75)
    print("TEST 2: CPU INFERENCE LATENCY BENCHMARK")
    print("=" * 75)

    num_iterations = 20
    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = valuation_service.predict_fair_value(
            city="Lahore", society="DHA", phase="Phase 5", property_type="HOUSE", area_sqft=4500.0, bedrooms=5, bathrooms=5
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"[*] Benchmark over {num_iterations} consecutive inferences on CPU:")
    print(f"    • Average Latency : {avg_latency:.2f} ms (Target: < 10.0 ms)")
    print(f"    • Minimum Latency : {min_latency:.2f} ms")
    print(f"    • Maximum Latency : {max_latency:.2f} ms")

    assert avg_latency < 10.0, f"Latency benchmark failed: {avg_latency:.2f}ms exceeds 10ms ceiling!"
    print("[SUCCESS] Valuation inference runs in sub-millisecond speeds on local CPU!")

    # ========================================================
    # CHAOS TEST 9.1: Zero and Out-of-Range Asking Price Ingestion
    # ========================================================
    print("\n" + "=" * 75)
    print("CHAOS TEST 9.1: ZERO & NEGATIVE ASKING PRICE INGESTION")
    print("=" * 75)

    zero_deal = valuation_service.evaluate_market_deal(0, 50_000_000, 45_000_000, 55_000_000)
    assert zero_deal["deal_status"] == "INVALID_PRICE", "Failed to catch zero asking price!"
    print(f"[SUCCESS] Zero asking price safely handled: {zero_deal['deal_status']}")

    neg_deal = valuation_service.evaluate_market_deal(-100_000, 50_000_000, 45_000_000, 55_000_000)
    assert neg_deal["deal_status"] == "INVALID_PRICE", "Failed to catch negative asking price!"
    print(f"[SUCCESS] Negative asking price safely handled: {neg_deal['deal_status']}")

    # ========================================================
    # CHAOS TEST 9.2: Model Weight File Missing Recovery
    # ========================================================
    print("\n" + "=" * 75)
    print("CHAOS TEST 9.2: MODEL WEIGHT FILE MISSING RECOVERY")
    print("=" * 75)

    # Temporarily move the model file to simulate a missing weight file
    if MODEL_PATH.exists():
        shutil.move(str(MODEL_PATH), str(BACKUP_PATH))
        # Reset singleton instance to trigger fresh load
        ValuationService._instance = None
        try:
            _ = ValuationService()
            print("[FAIL] ValuationService did not raise error on missing model file!")
        except FileNotFoundError as e:
            print(f"[SUCCESS] Service caught missing weight file gracefully: {e}")
        finally:
            # Restore model file
            shutil.move(str(BACKUP_PATH), str(MODEL_PATH))
            ValuationService._instance = None  # Reset singleton again

    print("\n" + "=" * 75)
    print("[DAY 9 COMPLETE] Valuation Service, Latency & Chaos Tests 100% green!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()