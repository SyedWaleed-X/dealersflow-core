import asyncio
from sqlalchemy import text
from backend.database import AsyncSessionLocal

async def test_dimension_mismatch():
    print("[*] Running Chaos Test 2.1: Injecting 512-dim vector into 384-dim column...")
    # Generate an invalid 512-dimension vector
    invalid_vector = str([0.1] * 512)
    
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text(f"""
                INSERT INTO properties (
                    intent, prop_type, city, society, price_pkr, area_sqft, embedding
                ) VALUES (
                    'SELL', 'PLOT', 'Lahore', 'DHA', 10000000, 2250, '{invalid_vector}'::vector
                );
            """))
            await session.commit()
            print("[FAIL] Database accepted invalid vector dimensions!")
        except Exception as e:
            print("[SUCCESS] Caught Expected PostgreSQL Boundary Error:")
            print(f"Error Type: {type(e).__name__}")
            print(f"Details: {str(e)[:120]}")

if __name__ == "__main__":
    asyncio.run(test_dimension_mismatch())