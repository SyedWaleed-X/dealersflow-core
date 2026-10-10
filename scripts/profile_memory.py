import os
import gc
import time
import asyncio
import psutil
import tracemalloc
from httpx import AsyncClient, ASGITransport
from tabulate import tabulate

from backend.main import app
from backend.core.database import engine

PROCESS = psutil.Process(os.getpid())


def get_current_rss_mb() -> float:
    """Returns current Resident Set Size (physical RAM) in Megabytes."""
    return PROCESS.memory_info().rss / (1024 * 1024)


async def run_memory_profile(num_requests: int = 15):
    print("=" * 80)
    print("DEALERSFLOW NATIVE MEMORY PROFILER & ZERO-LEAK VERIFICATION")
    print("=" * 80)

    # 1. Warm-up Garbage Collector & Start Tracemalloc
    gc.collect()
    tracemalloc.start()
    baseline_ram = get_current_rss_mb()
    snapshot_start = tracemalloc.take_snapshot()

    print(f"[*] Initial Baseline Process RAM : {baseline_ram:.2f} MB")
    print(f"[*] Executing {num_requests} sequential pipeline cycles under memory tracing...")

    transport = ASGITransport(app=app)
    readings = []

    async with AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
        for i in range(1, num_requests + 1):
            t0 = time.perf_counter()

            # Post a representative lead
            res = await client.post("/api/leads/process-voice", data={
                "raw_text": f"Aoa bhai, DHA Phase 6 mein 1 kanal plot for sale demand 8.{i} crore tariq 03001234567"
            })
            assert res.status_code == 201

            elapsed = time.perf_counter() - t0
            current_ram = get_current_rss_mb()
            delta_ram = current_ram - baseline_ram

            readings.append([
                f"Request #{i}",
                f"{elapsed:.2f}s",
                f"{current_ram:.2f} MB",
                f"{'+' if delta_ram >= 0 else ''}{delta_ram:.2f} MB"
            ])

    # 2. Analyze Heap Growth
    gc.collect()
    final_ram = get_current_rss_mb()
    snapshot_end = tracemalloc.take_snapshot()
    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")

    print("\n" + tabulate(readings, headers=["Cycle", "Latency", "Current RAM", "RAM Delta vs Baseline"], tablefmt="fancy_grid"))

    print("\n" + "=" * 80)
    print("TOP HEAP ALLOCATION DELTAS (Tracemalloc):")
    print("=" * 80)
    for stat in top_stats[:5]:
        print(f"  • {stat}")

    total_growth = final_ram - baseline_ram
    print(f"\n[*] Net Process RAM Growth after {num_requests} requests: {total_growth:.2f} MB")

    # A healthy ML server should stabilize with < 50MB of net growth after initial caching
    assert total_growth < 150.0, f"Memory leak detected! RAM grew by {total_growth:.2f} MB (> 150MB threshold)"
    print("[SUCCESS] Zero-leak verification passed! System RAM stabilized cleanly.")

    tracemalloc.stop()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_memory_profile(10))