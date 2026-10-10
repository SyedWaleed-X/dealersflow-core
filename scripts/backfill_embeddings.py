import asyncio
from sqlalchemy import text
from backend.core.database import AsyncSessionLocal, engine
from backend.ai.embedder import embedding_engine

BATCH_SIZE = 50

async def backfill():
    print("=" * 75)
    print("VECTOR EMBEDDING BACKFILLER (EMBEDDINGGEMMA 2 -> POSTGRESQL)")
    print("=" * 75)

    async with AsyncSessionLocal() as session:
        # Check how many properties need embeddings
        query_count = text("SELECT COUNT(*) FROM properties WHERE embedding IS NULL;")
        res = await session.execute(query_count)
        missing_count = res.scalar()
        print(f"[*] Found {missing_count} properties needing 768-dim vector embeddings...")

        if missing_count == 0:
            print("[+] All properties already have valid embeddings! Skipping.")
            return

        # Fetch properties needing embeddings
        fetch_query = text("""
            SELECT id, city, society, prop_type, raw_transcript, features->>'scraped_title' AS scraped_title 
            FROM properties 
            WHERE embedding IS NULL;
        """)
        rows = (await session.execute(fetch_query)).mappings().all()

        print(f"[*] Encoding {len(rows)} listings with EmbeddingGemma 2 in memory...")

        update_stmt = text("""
            UPDATE properties 
            SET embedding = CAST(:embedding AS vector)
            WHERE id = :id;
        """)

        updates = []
        for i, row in enumerate(rows, 1):
            # Assemble rich text description to capture maximum semantic context
            title = row["scraped_title"] or ""
            transcript = row["raw_transcript"] or ""
            text_to_encode = f"{row['city']} {row['society']} {row['prop_type']} {title} {transcript}".strip()

            vec = embedding_engine.embed_text(text_to_encode)
            updates.append({"id": row["id"], "embedding": str(vec)})

            if len(updates) >= BATCH_SIZE:
                await session.execute(update_stmt, updates)
                await session.commit()
                print(f"  [+] Committed batch: {i}/{len(rows)} properties embedded...")
                updates.clear()

        if updates:
            await session.execute(update_stmt, updates)
            await session.commit()
            print(f"  [+] Committed final batch: {len(rows)}/{len(rows)} embedded!")

    print("\n" + "=" * 75)
    print("[SUCCESS] All properties in PostgreSQL now indexed with 768-dim EmbeddingGemma 2 vectors!")
    print("=" * 75)
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(backfill())