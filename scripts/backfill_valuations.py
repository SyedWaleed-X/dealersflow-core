import asyncio
import json
from sqlalchemy import text
from backend.core.database import AsyncSessionLocal, engine
from backend.ai.valuation import valuation_service

async def backfill():
    print("=" * 75)
    print("CALCULATING CATBOOST APPRAISALS FOR ALL DATABASE INVENTORY")
    print("=" * 75)

    async with AsyncSessionLocal() as session:
        query = text("""
            SELECT id, city, society, prop_type, price_pkr, area_sqft, features
            FROM properties;
        """)
        rows = (await session.execute(query)).mappings().all()
        print(f"[*] Found {len(rows)} properties. Evaluating fair market values on CPU...")

        update_stmt = text("""
            UPDATE properties
            SET features = CAST(:features AS jsonb)
            WHERE id = :id;
        """)

        updated_count = 0
        for r in rows:
            city = r["city"]
            society = r["society"]
            prop_type = r["prop_type"]
            price_pkr = r["price_pkr"]
            area_sqft = r["area_sqft"]
            feat = r["features"] if isinstance(r["features"], dict) else json.loads(r["features"] or "{}")

            # Calculate CatBoost valuation
            val_pred = valuation_service.predict_fair_value(
                city=city,
                society=society,
                phase="General",
                property_type=prop_type,
                area_sqft=area_sqft
            )
            deal_eval = valuation_service.evaluate_market_deal(
                asking_price_pkr=price_pkr,
                estimated_price_pkr=val_pred["estimated_price_pkr"],
                lower_bound_pkr=val_pred["lower_bound_pkr"],
                upper_bound_pkr=val_pred["upper_bound_pkr"]
            )
            feat["valuation"] = {**val_pred, **deal_eval}

            await session.execute(update_stmt, {"id": r["id"], "features": json.dumps(feat)})
            updated_count += 1

        await session.commit()
        print(f"[SUCCESS] Updated {updated_count} properties with real CatBoost valuations & arbitrage spreads!")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(backfill())