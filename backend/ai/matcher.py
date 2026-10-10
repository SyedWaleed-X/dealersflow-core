import logging
from typing import List, Dict, Any
from sqlalchemy import text
from backend.core.database import AsyncSessionLocal
from backend.ai.embedder import embedding_engine

logger = logging.getLogger(__name__)


class LeadMatcher:
    """
    Two-Stage Hybrid Search Engine:
    Stage 1: Relational B-Tree Filters (Counter-Intent, City, Budget +/- Tolerance)
    Stage 2: 768-Dim HNSW Cosine Distance (<=>) against EmbeddingGemma 2 vectors.
    """

    @classmethod
    async def match_lead(
        cls,
        intent: str,                     # 'BUY' or 'SELL'
        city: str,                       # e.g., 'Lahore', 'Gujranwala', 'Islamabad'
        target_budget_pkr: int,          # Center budget
        tolerance_percentage: float = 0.25, # +/- 25% price flexibility
        preference_text: str = "",       # e.g., 'corner plot near mosque facing park'
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        # Counter-intent resolution (Buyers match with Sellers)
        counter_intent = "SELL" if intent.upper() == "BUY" else "BUY"

        min_price = int(target_budget_pkr * (1.0 - tolerance_percentage))
        max_price = int(target_budget_pkr * (1.0 + tolerance_percentage))

        # Generate 768-dim query vector using EmbeddingGemma 2
        search_query_text = f"{city} {preference_text}".strip()
        query_vector = str(embedding_engine.embed_text(search_query_text))

        query = text("""
            SELECT 
                id,
                city,
                society,
                prop_type,
                intent,
                price_pkr,
                area_sqft,
                features->>'listing_url' AS listing_url,
                features->>'posted_date' AS posted_date,
                features->>'scraped_title' AS title,
                (embedding <=> CAST(:query_vector AS vector)) AS cosine_distance,
                ROUND(CAST((1 - (embedding <=> CAST(:query_vector AS vector))) AS numeric), 4) AS similarity_score
            FROM properties
            WHERE LOWER(city) = LOWER(:city)
              AND intent = CAST(:counter_intent AS property_intent)
              AND price_pkr BETWEEN :min_price AND :max_price
            ORDER BY embedding <=> CAST(:query_vector AS vector) ASC
            LIMIT :limit;
        """)

        async with AsyncSessionLocal() as session:
            result = await session.execute(query, {
                "city": city,
                "counter_intent": counter_intent,
                "min_price": min_price,
                "max_price": max_price,
                "query_vector": query_vector,
                "limit": limit
            })

            rows = result.mappings().all()

            matches = []
            for row in rows:
                matches.append({
                    "id": str(row["id"]),
                    "title": row["title"],
                    "city": row["city"],
                    "society": row["society"],
                    "prop_type": row["prop_type"],
                    "price_pkr": row["price_pkr"],
                    "area_sqft": row["area_sqft"],
                    "listing_url": row["listing_url"],
                    "posted_date": row["posted_date"],
                    "cosine_distance": float(row["cosine_distance"]) if row["cosine_distance"] is not None else 1.0,
                    "similarity_score": float(row["similarity_score"]) if row["similarity_score"] is not None else 0.0
                })

            return matches