import asyncio
import json
import random
import numpy as np
from sqlalchemy import text
from backend.database import engine, AsyncSessionLocal

CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi"]
INTENTS = ["BUY", "SELL"]
PROP_TYPES = ["PLOT", "HOUSE", "COMMERCIAL"]

SOCIETIES_BY_CITY = {
    "Lahore": ["DHA Phase 6", "DHA Phase 5", "Bahria Town", "Gulberg III", "Johar Town"],
    "Karachi": ["DHA Phase 8", "Clifton", "Bahria Town Karachi", "Gulshan-e-Iqbal"],
    "Islamabad": ["DHA Phase 2", "Bahria Town Phase 7", "Sector F-7", "Sector G-11"],
    "Rawalpindi": ["Bahria Town Phase 4", "Chaklala Scheme 3", "Satellite Town"]
}

SECTORS = ["Block A", "Block B", "Block CCA", "Sector C", "Sector J"]

FEATURES_POOL = [
    {"corner": True, "park_facing": True},
    {"corner": False, "park_facing": True, "near_mosque": True},
    {"commercial_access": True, "main_boulevard": True},
    {"possession_paid": True, "utilities_available": True},
    {"standard_plot": True}
]

def generate_random_normalized_vector(dim: int = 384) -> list[float]:
    """Generates a unit-normalized random vector to simulate MiniLM-L6-v2 embeddings."""
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

async def seed_database(count: int = 500):
    print(f"[*] Seeding {count} realistic property listings into PostgreSQL...")
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM properties;"))
        initial_count = result.scalar()
        print(f"[*] Initial row count in 'properties': {initial_count}")

        records = []
        for _ in range(count):
            city = random.choice(CITIES)
            society = random.choice(SOCIETIES_BY_CITY[city])
            intent = random.choice(INTENTS)
            prop_type = random.choice(PROP_TYPES)
            
            # Realistic price generation (50 Lacs to 12 Crore PKR)
            price_pkr = random.randint(50, 1200) * 100_000
            
            # Standard LDA area: 5 Marla (1125 sqft), 10 Marla (2250 sqft), 1 Kanal (4500 sqft)
            area_sqft = float(random.choice([1125, 2250, 4500, 9000]))
            
            bedrooms = 0 if prop_type == "PLOT" else random.randint(2, 6)
            bathrooms = 0 if prop_type == "PLOT" else bedrooms + random.choice([0, 1])
            
            embedding_vec = generate_random_normalized_vector(384)
            features = random.choice(FEATURES_POOL)

            records.append({
                "intent": intent,
                "prop_type": prop_type,
                "city": city,
                "society": society,
                "phase": f"Phase {random.randint(1, 8)}",
                "sector_block": random.choice(SECTORS),
                "price_pkr": price_pkr,
                "area_sqft": area_sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "raw_transcript": f"Simulated voice note for {prop_type} in {society}, {city}.",
                # Standard json.dumps ensures booleans become lowercase 'true'/'false'
                "features": json.dumps(features),
                "contact_name": f"Dealer {random.randint(100, 999)}",
                "contact_phone": f"+92300{random.randint(1000000, 9999999)}",
                "embedding": str(embedding_vec)
            })

        # ANSI CAST with bind parameters for bulletproof asyncpg execution
        insert_query = text("""
            INSERT INTO properties (
                intent, prop_type, city, society, phase, sector_block,
                price_pkr, area_sqft, bedrooms, bathrooms, raw_transcript,
                features, contact_name, contact_phone, embedding
            ) VALUES (
                CAST(:intent AS property_intent),
                CAST(:prop_type AS property_type),
                :city, :society, :phase, :sector_block,
                :price_pkr, :area_sqft, :bedrooms, :bathrooms, :raw_transcript,
                CAST(:features AS jsonb),
                :contact_name, :contact_phone,
                CAST(:embedding AS vector)
            );
        """)

        await session.execute(insert_query, records)
        await session.commit()
        
        result = await session.execute(text("SELECT COUNT(*) FROM properties;"))
        final_count = result.scalar()
        print(f"[+] Seeding successful! Total rows in 'properties': {final_count}")

async def run_explain_analyze():
    print("\n[*] Running EXPLAIN ANALYZE on Hybrid Query (Hard Relational + Vector Cosine)...")
    
    query_vector = str(generate_random_normalized_vector(384))
    
    # Query: Filter by Lahore, SELL, and price range, then order by cosine similarity (<=>)
    explain_query = text(f"""
        EXPLAIN ANALYZE
        SELECT id, city, society, price_pkr, (embedding <=> '{query_vector}'::vector) AS distance
        FROM properties
        WHERE city = 'Lahore'
          AND intent = 'SELL'
          AND price_pkr BETWEEN 20000000 AND 60000000
        ORDER BY distance
        LIMIT 5;
    """)

    async with AsyncSessionLocal() as session:
        result = await session.execute(explain_query)
        rows = result.fetchall()
        print("-" * 75)
        for row in rows:
            print(row[0])
        print("-" * 75)

async def main():
    await seed_database(500)
    await run_explain_analyze()
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())