import os
import sys
import re
import json
import asyncio
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from starlette.requests import Request
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import engine, get_db
from backend.ai.agent_graph import agent_pipeline
from backend.ai.graph_state import AgentPipelineState
from backend.ai.embedder import embedding_engine

logger = logging.getLogger("dealersflow.api")
logging.basicConfig(level=logging.INFO)


# ==========================================
# MODERN LIFESPAN MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[*] FastAPI booting: Validating PostgreSQL connection & warming models...")
    try:
        async with engine.begin() as conn:
            res = await conn.execute(text("SELECT count(*) FROM properties;"))
            count = res.scalar()
            logger.info(f"[+] PostgreSQL verified! Active property records: {count}")

        _ = embedding_engine.embed_text("startup warmup")
        logger.info("[+] EmbeddingGemma 2 warmed up in memory.")

    except Exception as e:
        logger.error(f"[!] Database startup check failed: {e}")

    yield

    logger.info("[*] FastAPI shutting down: Disposing connection pool...")
    await engine.dispose()
    logger.info("[+] Database connections cleanly closed.")


app = FastAPI(
    title="DealersFlow Core AI API",
    description="Real Estate Multi-Agent CRM Engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from backend.core.security import PayloadSizeLimitMiddleware

# Add Defensive Security Middleware
app.add_middleware(PayloadSizeLimitMiddleware)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "compute_profile": settings.COMPUTE_PROFILE,
        "intake_provider": settings.INTAKE_LLM_PROVIDER,
        "batch_provider": settings.BATCH_LLM_PROVIDER
    }


# ==========================================
# 1. PROCESS VOICE / TEXT LEAD ENDPOINT
# ==========================================
@app.post("/api/leads/process-voice", status_code=status.HTTP_201_CREATED)
async def process_voice_lead(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    if not file and not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an audio file or raw_text must be provided."
        )

    temp_audio_path = None

    try:
        if file:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(file.filename or "voice.ogg")[1]
            )
            temp_audio_path = temp_file.name

            chunk_size = 1024 * 1024
            while chunk := await file.read(chunk_size):
                temp_file.write(chunk)
            temp_file.close()

            logger.info(f"[*] Audio safely buffered to disk: {temp_audio_path}")

        initial_state: AgentPipelineState = {
            "audio_path": temp_audio_path,
            "raw_text": raw_text,
            "transcript": "",
            "extracted_lead": None,
            "normalized_lead": None,
            "valuation_result": None,
            "matched_leads": [],
            "whatsapp_message": "",
            "whatsapp_deep_link": "",
            "error": None,
            "is_success": False
        }

        final_state = await agent_pipeline.ainvoke(initial_state)

        if not final_state.get("is_success"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Pipeline processing failed: {final_state.get('error')}"
            )

        norm_lead = final_state.get("normalized_lead", {})
        val_res = final_state.get("valuation_result", {})

        text_for_embedding = f"{norm_lead.get('city')} {norm_lead.get('society')} {norm_lead.get('property_type')} {final_state.get('transcript')}"
        vector = embedding_engine.embed_text(text_for_embedding)

        features_data = {
            "valuation": val_res,
            "transcript": final_state.get("transcript"),
            "whatsapp_link": final_state.get("whatsapp_deep_link"),
            "data_source": "voice_ai" if file else "manual_dealer",
            "qualitative_features": norm_lead.get("qualitative_features", [])
        }

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
            ) RETURNING id;
        """)

        result = await db.execute(insert_query, {
            "intent": norm_lead.get("intent", "BUY"),
            "prop_type": norm_lead.get("property_type", "PLOT"),
            "city": norm_lead.get("city", "Islamabad"),
            "society": norm_lead.get("society", "General Area"),
            "phase": norm_lead.get("phase"),
            "sector_block": norm_lead.get("sector_block"),
            "price_pkr": norm_lead.get("price_pkr", 15_000_000),
            "area_sqft": norm_lead.get("area_sqft", 1125.0),
            "bedrooms": norm_lead.get("bedrooms", 0),
            "bathrooms": norm_lead.get("bathrooms", 0),
            "raw_transcript": final_state.get("transcript"),
            "features": json.dumps(features_data),
            "contact_name": norm_lead.get("client_name"),
            "contact_phone": norm_lead.get("client_phone"),
            "embedding": str(vector)
        })
        await db.commit()

        created_id = str(result.scalar())

        for m in final_state.get("matched_leads", []):
            try:
                await db.execute(text("""
                    INSERT INTO lead_matches (buyer_id, seller_id, similarity_score, match_status)
                    VALUES (CAST(:b_id AS uuid), CAST(:s_id AS uuid), :score, 'NEW')
                    ON CONFLICT (buyer_id, seller_id) DO NOTHING;
                """), {
                    "b_id": created_id if norm_lead.get("intent") == "BUY" else m["id"],
                    "s_id": m["id"] if norm_lead.get("intent") == "BUY" else created_id,
                    "score": m.get("similarity_score", 0.85)
                })
            except Exception:
                pass
        await db.commit()

        return {
            "property_id": created_id,
            "transcript": final_state.get("transcript"),
            "lead": norm_lead,
            "valuation": val_res,
            "matched_leads_count": len(final_state.get("matched_leads", [])),
            "matched_leads": final_state.get("matched_leads", []),
            "whatsapp_message": final_state.get("whatsapp_message"),
            "whatsapp_deep_link": final_state.get("whatsapp_deep_link")
        }

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# ==========================================
# 2. MANUAL LEAD INTAKE ENDPOINT
# ==========================================
class ManualLeadRequest(BaseModel):
    city: str
    society: str
    property_type: str = "PLOT"
    intent: str = "BUY"
    price_pkr: int = 15000000
    area_sqft: float = 1125.0
    client_name: Optional[str] = "Client"
    client_phone: Optional[str] = "923001234567"
    notes: Optional[str] = ""

@app.post("/api/leads/manual", status_code=status.HTTP_201_CREATED)
async def create_manual_lead(
    payload: ManualLeadRequest,
    db: AsyncSession = Depends(get_db)
):
    from backend.ai.valuation import valuation_service
    from backend.ai.matcher import LeadMatcher
    from backend.services.whatsapp import whatsapp_service

    val_pred = valuation_service.predict_fair_value(
        city=payload.city,
        society=payload.society,
        phase="General",
        property_type=payload.property_type,
        area_sqft=payload.area_sqft
    )
    deal_eval = valuation_service.evaluate_market_deal(
        asking_price_pkr=payload.price_pkr,
        estimated_price_pkr=val_pred["estimated_price_pkr"],
        lower_bound_pkr=val_pred["lower_bound_pkr"],
        upper_bound_pkr=val_pred["upper_bound_pkr"]
    )
    valuation_data = {**val_pred, **deal_eval}

    text_corpus = f"{payload.city} {payload.society} {payload.property_type} {payload.notes}"
    vec = embedding_engine.embed_text(text_corpus)

    search_budget = payload.price_pkr if payload.price_pkr > 0 else val_pred["estimated_price_pkr"]
    matched = await LeadMatcher.match_lead(
        intent=payload.intent,
        city=payload.city,
        target_budget_pkr=search_budget,
        tolerance_percentage=0.35,
        preference_text=f"{payload.society} {payload.notes}",
        limit=5
    )

    msg = whatsapp_service.generate_negotiation_template(
        client_name=payload.client_name,
        property_type=payload.property_type,
        society=payload.society,
        city=payload.city,
        asking_price_pkr=payload.price_pkr,
        valuation_dict=valuation_data,
        matched_leads=matched
    )
    wa_app_link = whatsapp_service.generate_wa_link(payload.client_phone, msg)
    wa_web_link = whatsapp_service.generate_wa_web_link(payload.client_phone, msg)



    features_dict = {
        "valuation": valuation_data,
        "whatsapp_app_link": wa_app_link,
        "whatsapp_web_link": wa_web_link,
        "data_source": "manual_dealer",
        "scraped_title": f"{payload.property_type} in {payload.society}, {payload.city}"
    }

    insert_stmt = text("""
        INSERT INTO properties (
            intent, prop_type, city, society, price_pkr, area_sqft,
            raw_transcript, features, contact_name, contact_phone, embedding
        ) VALUES (
            CAST(:intent AS property_intent),
            CAST(:prop_type AS property_type),
            :city, :society, :price_pkr, :area_sqft,
            :raw_transcript, CAST(:features AS jsonb),
            :contact_name, :contact_phone, CAST(:embedding AS vector)
        ) RETURNING id;
    """)

    res = await db.execute(insert_stmt, {
        "intent": payload.intent,
        "prop_type": payload.property_type,
        "city": payload.city,
        "society": payload.society,
        "price_pkr": payload.price_pkr,
        "area_sqft": payload.area_sqft,
        "raw_transcript": payload.notes or f"Manual Lead for {payload.property_type}",
        "features": json.dumps(features_dict),
        "contact_name": payload.client_name,
        "contact_phone": payload.client_phone,
        "embedding": str(vec)
    })
    await db.commit()

    created_id = str(res.scalar())

    for m in matched:
        try:
            await db.execute(text("""
                INSERT INTO lead_matches (buyer_id, seller_id, similarity_score, match_status)
                VALUES (CAST(:b_id AS uuid), CAST(:s_id AS uuid), :score, 'NEW')
                ON CONFLICT (buyer_id, seller_id) DO NOTHING;
            """), {
                "b_id": created_id if payload.intent == "BUY" else m["id"],
                "s_id": m["id"] if payload.intent == "BUY" else created_id,
                "score": m.get("similarity_score", 0.85)
            })
        except Exception:
            pass
    await db.commit()

    return {
        "property_id": created_id,
        "lead": payload.model_dump(),
        "valuation": valuation_data,
        "matched_leads_count": len(matched),
        "matched_leads": matched,
        "whatsapp_app_link": wa_app_link,
        "whatsapp_web_link": wa_web_link,
        "whatsapp_message": msg
    }


# ==========================================
# 3. GLOBAL BILATERAL MATCHES FEED (NULL-SAFE)
# ==========================================
@app.get("/api/matches/feed", status_code=status.HTTP_200_OK)
async def get_matches_feed(
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    from backend.ai.matcher import LeadMatcher

    buyers_query = text("""
        SELECT id, city, society, prop_type, price_pkr, area_sqft, raw_transcript, 
               COALESCE(features->>'data_source', features->>'verified_source', features->>'source', 'manual_dealer') AS data_source,
               contact_name, contact_phone
        FROM properties
        WHERE intent = 'BUY'
        ORDER BY created_at DESC
        LIMIT 10;
    """)
    buyer_rows = (await db.execute(buyers_query)).mappings().all()

    match_pairings = []

    for buyer in buyer_rows:
        buyer_source = str(buyer["data_source"]).lower()
        matched_sellers = await LeadMatcher.match_lead(
            intent="BUY",
            city=buyer["city"],
            target_budget_pkr=buyer["price_pkr"] if buyer["price_pkr"] > 0 else 15000000,
            tolerance_percentage=0.35,
            preference_text=f"{buyer['society']} {buyer['raw_transcript']}",
            limit=2
        )

        for seller in matched_sellers:
            seller_link = str(seller.get("listing_url") or "").lower()
            if "zameen" in seller_link:
                match_type = f"{buyer_source.replace('_', ' ').title()} ↔ Zameen"
            elif "olx" in seller_link:
                match_type = f"{buyer_source.replace('_', ' ').title()} ↔ OLX"
            else:
                match_type = f"{buyer_source.replace('_', ' ').title()} ↔ Verified Listing"

            match_pairings.append({
                "match_id": f"{buyer['id']}-{seller['id']}",
                "match_type": match_type,
                "similarity_score": seller.get("similarity_score", 0.82),
                "buyer": {
                    "id": str(buyer["id"]),
                    "city": buyer["city"],
                    "society": buyer["society"],
                    "prop_type": buyer["prop_type"],
                    "budget_pkr": buyer["price_pkr"],
                    "area_sqft": buyer["area_sqft"],
                    "transcript": buyer["raw_transcript"],
                    "source": buyer_source,
                    "contact_phone": buyer["contact_phone"] or "923001234567"
                },
                "seller": {
                    "id": seller["id"],
                    "city": seller["city"],
                    "society": seller["society"],
                    "prop_type": seller["prop_type"],
                    "price_pkr": seller["price_pkr"],
                    "area_sqft": seller["area_sqft"],
                    "title": seller.get("title") or f"{seller['prop_type']} in {seller['society']}",
                    "listing_url": seller.get("listing_url"),
                    "posted_date": seller.get("posted_date", "Active")
                }
            })

    return {
        "total_pairs": len(match_pairings),
        "pairings": match_pairings[:limit]
    }


# ==========================================
# 4. RUN BATCH AI MATCHMAKER
# ==========================================
@app.post("/api/matches/run", status_code=status.HTTP_200_OK)
async def run_batch_matchmaker(
    db: AsyncSession = Depends(get_db)
):
    from backend.ai.matcher import LeadMatcher

    buyers_query = text("""
        SELECT id, city, society, prop_type, price_pkr, area_sqft, raw_transcript 
        FROM properties 
        WHERE intent = 'BUY'
        ORDER BY created_at DESC;
    """)
    buyers = (await db.execute(buyers_query)).mappings().all()

    inserted_new_matches = 0

    for buyer in buyers:
        matched_sellers = await LeadMatcher.match_lead(
            intent="BUY",
            city=buyer["city"],
            target_budget_pkr=buyer["price_pkr"] if buyer["price_pkr"] > 0 else 15000000,
            tolerance_percentage=0.35,
            preference_text=f"{buyer['society']} {buyer['raw_transcript']}",
            limit=5
        )

        for seller in matched_sellers:
            insert_match_stmt = text("""
                INSERT INTO lead_matches (buyer_id, seller_id, similarity_score, match_status)
                VALUES (CAST(:buyer_id AS uuid), CAST(:seller_id AS uuid), :score, 'NEW')
                ON CONFLICT (buyer_id, seller_id) DO NOTHING;
            """)
            res = await db.execute(insert_match_stmt, {
                "buyer_id": buyer["id"],
                "seller_id": seller["id"],
                "score": seller.get("similarity_score", 0.85)
            })
            if res.rowcount > 0:
                inserted_new_matches += 1

    await db.commit()
    return {
        "status": "success",
        "buyers_scanned": len(buyers),
        "new_matches_discovered": inserted_new_matches
    }


# ==========================================
# 5. ON-DEMAND MODULAR SCRAPER
# ==========================================
class ScrapeTriggerRequest(BaseModel):
    city: str = "Gujranwala"
    portal: str = "OLX"
    max_items: int = 8

SCRAPER_URL_REGISTRY = {
    "OLX": {
        "Gujranwala": "https://www.olx.com.pk/gujranwala_g4060662/properties_c2",
        "Lahore": "https://www.olx.com.pk/lahore_g4060673/properties_c2",
        "Islamabad": "https://www.olx.com.pk/islamabad_g4060615/properties_c2",
        "Karachi": "https://www.olx.com.pk/karachi_g4060695/properties_c2",
    },
    "Zameen": {
        "Lahore": "https://www.zameen.com/Homes/Lahore-1-1.html",
        "Islamabad": "https://www.zameen.com/Homes/Islamabad-3-1.html",
    }
}


def _execute_scraper_in_isolated_proactor(city: str, portal: str, max_items: int) -> List[Dict[str, Any]]:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    async def _async_worker():
        from backend.scrapers.olx import OLXScraper
        from backend.scrapers.zameen import ZameenScraper

        olx_scraper = OLXScraper()
        zameen_scraper = ZameenScraper()
        items = []

        portals_to_run = ["OLX", "Zameen"] if portal == "All" else [portal]

        for p_name in portals_to_run:
            registry = SCRAPER_URL_REGISTRY.get(p_name, {})
            cities_to_run = list(registry.keys()) if city == "All" else [city]

            for c_name in cities_to_run:
                if c_name in registry:
                    url = registry[c_name]
                    scraper = olx_scraper if p_name == "OLX" else zameen_scraper
                    res = await scraper.scrape(url, max_items=max_items)
                    for r in res:
                        r["target_city"] = c_name
                        items.append(r)

        return items

    return asyncio.run(_async_worker())


@app.post("/api/scraper/trigger", status_code=status.HTTP_200_OK)
async def trigger_modular_scrape(
    payload: ScrapeTriggerRequest,
    db: AsyncSession = Depends(get_db)
):
    from backend.domain.normalizer import PropertyNormalizer

    collected_items = await run_in_threadpool(
        _execute_scraper_in_isolated_proactor,
        payload.city,
        payload.portal,
        payload.max_items
    )

    saved_new = 0
    duplicates_skipped = 0
    fresh_saved_listings = []

    for item in collected_items:
        title = item.get("title", "Listing")
        raw_price = item.get("raw_price", "")
        city = item.get("target_city", "Gujranwala")
        society = item.get("location", "General Area")
        transcript = item.get("raw_transcript", title)
        listing_url = item.get("listing_url", "")
        posted_date = item.get("posted_date", "Recently")
        source = "olx_pakistan" if "olx" in str(item.get("source")).lower() else "zameen_com"

        price_pkr = PropertyNormalizer.normalize_price(raw_price) or 15_000_000
        area_sqft = PropertyNormalizer.normalize_area(transcript) or 1125.0

        prop_type = "PLOT"
        if any(k in (title + " " + transcript).lower() for k in ["house", "makan", "kothi", "flat"]):
            prop_type = "HOUSE"
        elif any(k in (title + " " + transcript).lower() for k in ["commercial", "shop", "plaza"]):
            prop_type = "COMMERCIAL"

        # Compute CatBoost valuation and arbitrage rating for scraped listings
        from backend.ai.valuation import valuation_service
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
        val_payload = {**val_pred, **deal_eval}

        features_dict = {
            "listing_url": listing_url,
            "posted_date": posted_date,
            "scraped_title": title,
            "data_source": source,
            "valuation": val_payload
        }

        stmt = text("""
            INSERT INTO properties (
                intent, prop_type, city, society, price_pkr, area_sqft, 
                raw_transcript, features, listing_url
            ) VALUES (
                'SELL'::property_intent,
                CAST(:prop_type AS property_type),
                :city, :society, :price_pkr, :area_sqft,
                :raw_transcript, CAST(:features AS jsonb), :listing_url
            )
            ON CONFLICT (listing_url) DO NOTHING
            RETURNING id;
        """)

        res = await db.execute(stmt, {
            "prop_type": prop_type,
            "city": city,
            "society": society,
            "price_pkr": price_pkr,
            "area_sqft": area_sqft,
            "raw_transcript": transcript,
            "features": json.dumps(features_dict),
            "listing_url": listing_url
        })
        row = res.fetchone()
        if row:
            saved_new += 1
            fresh_saved_listings.append({
                "id": str(row[0]),
                "title": title,
                "city": city,
                "society": society,
                "price_pkr": price_pkr,
                "listing_url": listing_url,
                "posted_date": posted_date,
                "data_source": source
            })
        else:
            duplicates_skipped += 1

    await db.commit()

    # Automatically trigger matchmaker on newly acquired inventory
    matches_res = await run_batch_matchmaker(db)

    return {
        "status": "success",
        "city": payload.city,
        "portal": payload.portal,
        "scraped_count": len(collected_items),
        "saved_new": saved_new,
        "duplicates_skipped": duplicates_skipped,
        "new_matches_created": matches_res.get("new_matches_discovered", 0),
        "fresh_listings": fresh_saved_listings
    }


# ==========================================
# 6. INVENTORY QUERY
# ==========================================
@app.get("/api/leads", status_code=status.HTTP_200_OK)
async def get_leads(
    city: Optional[str] = Query(None),
    intent: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    where_clauses = []
    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if city and city != "All":
        where_clauses.append("LOWER(city) = LOWER(:city)")
        params["city"] = city
    if intent and intent != "All":
        where_clauses.append("intent = CAST(:intent AS property_intent)")
        params["intent"] = intent.upper()

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = text(f"""
        SELECT 
            id, city, society, prop_type, intent, price_pkr, area_sqft,
            raw_transcript,
            features->'valuation' AS valuation,
            created_at, 
            features->>'listing_url' AS listing_url,
            features->>'posted_date' AS posted_date,
            features->>'scraped_title' AS title,
            COALESCE(features->>'data_source', features->>'verified_source', features->>'source', 'manual_dealer') AS data_source,
            contact_phone
        FROM properties
        {where_sql}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset;
    """)

    count_query = text(f"SELECT COUNT(*) FROM properties {where_sql};")

    rows = (await db.execute(query, params)).mappings().all()
    total_count = (await db.execute(count_query, params)).scalar()

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "items": [dict(r) for r in rows]
    }




# ==========================================
# 7. UPDATE PROPERTY PHONE NUMBER IN DB
# ==========================================
class PropertyPhoneUpdate(BaseModel):
    property_id: str
    phone: str

@app.post("/api/properties/update-phone", status_code=status.HTTP_200_OK)
async def update_property_phone(
    payload: PropertyPhoneUpdate,
    db: AsyncSession = Depends(get_db)
):
    from backend.services.whatsapp import whatsapp_service
    clean_phone = whatsapp_service.sanitize_phone_number(payload.phone)

    stmt = text("""
        UPDATE properties 
        SET contact_phone = :phone 
        WHERE id = CAST(:id AS uuid);
    """)
    await db.execute(stmt, {"id": payload.property_id, "phone": clean_phone})
    await db.commit()
    return {"status": "success", "clean_phone": clean_phone}


# ==========================================
# 8. BATCH WHATSAPP GROUP CHAT PARSER
# ==========================================
class BatchChatRequest(BaseModel):
    chat_log: str

WHATSAPP_LINE_REGEX = re.compile(
    r"(?:\[?(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)[,\s\d/]*\]?\s*(?:-\s*)?)([^:]+):\s*(.*)"
)
REAL_ESTATE_KEYWORDS = re.compile(
    r"\b(plot|marla|kanal|crore|cr|lac|lakh|bechna|chahiye|demand|sqft|house|makan|flat|shop|plaza)\b",
    re.IGNORECASE
)

# ==========================================
# 9. AUTONOMOUS WEB SCOUT & HUNTER API
# ==========================================
# ==========================================
# 9. AUTONOMOUS WEB HUNTER ENDPOINTS
# ==========================================
class WebHunterRequest(BaseModel):
    buyer_id: Optional[str] = None
    city: str = "Gujranwala"
    society: str = "Citi Housing"
    property_type: str = "PLOT"
    budget_pkr: int = 15000000
    search_notes: Optional[str] = ""
    duration_seconds: int = 60

@app.post("/api/hunter/scout", status_code=status.HTTP_200_OK)
async def trigger_web_hunter(
    payload: WebHunterRequest,
    db: AsyncSession = Depends(get_db)
):
    from backend.ai.web_hunter import AutonomousWebHunter

    result = await AutonomousWebHunter.hunt_for_lead(
        buyer_lead_id=payload.buyer_id,
        city=payload.city,
        society=payload.society,
        property_type=payload.property_type,
        budget_pkr=payload.budget_pkr,
        search_notes=payload.search_notes or "",
        duration_seconds=payload.duration_seconds
    )
    return result

@app.post("/api/hunter/stop", status_code=status.HTTP_200_OK)
async def stop_web_hunter():
    """Immediately stops all background searching loops."""
    from backend.ai.web_hunter import AutonomousWebHunter
    AutonomousWebHunter.request_cancel()
    return {"status": "stopped"}

@app.get("/api/hunter/findings", status_code=status.HTTP_200_OK)
async def get_all_scout_findings(
    db: AsyncSession = Depends(get_db)
):
    """Returns all scouted findings grouped by buyer lead."""
    stmt = text("""
        SELECT 
            w.id AS scout_id, w.title, w.url, w.snippet, w.similarity_score, w.price_pkr, w.created_at,
            p.id AS buyer_id, p.city, p.society, p.prop_type, p.price_pkr AS buyer_budget, p.raw_transcript
        FROM web_scout_results w
        LEFT JOIN properties p ON w.buyer_id = p.id
        ORDER BY w.created_at DESC;
    """)
    rows = (await db.execute(stmt)).mappings().all()
    return {"total": len(rows), "findings": [dict(r) for r in rows]}

class ImportScoutRequest(BaseModel):
    scout_id: str
    city: str = "Gujranwala"
    society: str = "General"
    prop_type: str = "PLOT"
    price_pkr: int = 15000000
    title: str
    url: str
    snippet: str

@app.post("/api/hunter/import", status_code=status.HTTP_201_CREATED)
async def import_scout_to_inventory(
    payload: ImportScoutRequest,
    db: AsyncSession = Depends(get_db)
):
    """Allows agent to promote a scouted web deal directly into the permanent property inventory."""
    features_dict = {
        "listing_url": payload.url,
        "scraped_title": payload.title,
        "data_source": "imported_from_scout",
        "posted_date": "Imported Web Deal"
    }

    vec = embedding_engine.embed_text(f"{payload.city} {payload.society} {payload.prop_type} {payload.title}")

    stmt = text("""
        INSERT INTO properties (
            intent, prop_type, city, society, price_pkr, area_sqft, 
            raw_transcript, features, listing_url, embedding
        ) VALUES (
            'SELL'::property_intent,
            CAST(:prop_type AS property_type),
            :city, :society, :price_pkr, 2250.0,
            :raw_transcript, CAST(:features AS jsonb), :listing_url,
            CAST(:embedding AS vector)
        )
        ON CONFLICT (listing_url) DO NOTHING
        RETURNING id;
    """)

    res = await db.execute(stmt, {
        "prop_type": payload.prop_type,
        "city": payload.city,
        "society": payload.society,
        "price_pkr": payload.price_pkr,
        "raw_transcript": payload.snippet,
        "features": json.dumps(features_dict),
        "listing_url": payload.url,
        "embedding": str(vec)
    })
    await db.commit()
    row = res.fetchone()
    return {"status": "imported", "property_id": str(row[0]) if row else None}
# ==========================================
# 9. FAIL-SAFE DEMO FIXTURES ENDPOINT
# ==========================================
@app.get("/api/demo/fixtures", status_code=status.HTTP_200_OK)
async def get_demo_fixtures():
    """Returns pre-configured golden demonstration leads for presentation reliability."""
    from backend.core.demo_fixtures import GOLDEN_DEMO_FIXTURES
    return {"fixtures": GOLDEN_DEMO_FIXTURES}

@app.post("/api/leads/batch-chat", status_code=status.HTTP_201_CREATED)
async def parse_batch_whatsapp_chat(
    payload: BatchChatRequest,
    db: AsyncSession = Depends(get_db)
):
    from backend.ai.extractor import LeadExtractor
    from backend.domain.schemas import NormalizedPropertyLead, ExtractedPropertyLead

    extractor = LeadExtractor()
    lines = payload.chat_log.split("\n")
    parsed_leads = []

    # If Gemini is configured, use its 1M context window for single-pass dissection
    if settings.BATCH_LLM_PROVIDER.value == "GEMINI" and settings.GEMINI_API_KEY:
        raw_items = extractor.extract_batch_chat(payload.chat_log)
        for item in raw_items:
            try:
                extracted = ExtractedPropertyLead(**item) if isinstance(item, dict) else item
                normalized = NormalizedPropertyLead.from_extracted_lead(extracted)

                features_data = {
                    "scraped_title": f"{normalized.property_type.value.title()} Inquiry via WhatsApp Group Dump",
                    "data_source": "whatsapp_group_dump",
                    "sender_name": extracted.client_name or "Dealer"
                }

                stmt = text("""
                    INSERT INTO properties (
                        intent, prop_type, city, society, price_pkr, area_sqft,
                        raw_transcript, features, contact_name, contact_phone
                    ) VALUES (
                        CAST(:intent AS property_intent),
                        CAST(:prop_type AS property_type),
                        :city, :society, :price_pkr, :area_sqft,
                        :raw_transcript, CAST(:features AS jsonb), :contact_name, :contact_phone
                    ) RETURNING id;
                """)

                res = await db.execute(stmt, {
                    "intent": normalized.intent,
                    "prop_type": normalized.property_type,
                    "city": normalized.city,
                    "society": normalized.society,
                    "price_pkr": normalized.price_pkr,
                    "area_sqft": normalized.area_sqft,
                    "raw_transcript": f"Group chat lead from {extracted.client_name}",
                    "features": json.dumps(features_data),
                    "contact_name": extracted.client_name or "Dealer",
                    "contact_phone": extracted.client_phone or "923001234567"
                })
                lead_id = str(res.scalar())
                parsed_leads.append({"id": lead_id, "lead": normalized.model_dump()})
            except Exception as e:
                logger.warning(f"[!] Error processing batch item: {e}")
    else:
        # Local Line-by-Line Sieve
        for line in lines:
            match = WHATSAPP_LINE_REGEX.search(line)
            if match:
                sender = match.group(2).strip()
                message_text = match.group(3).strip()
            else:
                sender = "Dealer"
                message_text = line.strip()

            if not REAL_ESTATE_KEYWORDS.search(message_text):
                continue

            try:
                extracted = extractor.extract(message_text)
                extracted.client_name = sender
                normalized = NormalizedPropertyLead.from_extracted_lead(extracted)

                features_data = {
                    "scraped_title": f"{normalized.property_type} inquiry via WhatsApp Group",
                    "data_source": "whatsapp_group_dump",
                    "sender_name": sender
                }

                stmt = text("""
                    INSERT INTO properties (
                        intent, prop_type, city, society, price_pkr, area_sqft,
                        raw_transcript, features, contact_name, contact_phone
                    ) VALUES (
                        CAST(:intent AS property_intent),
                        CAST(:prop_type AS property_type),
                        :city, :society, :price_pkr, :area_sqft,
                        :raw_transcript, CAST(:features AS jsonb), :contact_name, :contact_phone
                    ) RETURNING id;
                """)

                res = await db.execute(stmt, {
                    "intent": normalized.intent,
                    "prop_type": normalized.property_type,
                    "city": normalized.city,
                    "society": normalized.society,
                    "price_pkr": normalized.price_pkr,
                    "area_sqft": normalized.area_sqft,
                    "raw_transcript": message_text,
                    "features": json.dumps(features_data),
                    "contact_name": sender,
                    "contact_phone": "923001234567"
                })
                lead_id = str(res.scalar())
                parsed_leads.append({"id": lead_id, "sender": sender, "lead": normalized.model_dump()})
            except Exception as e:
                logger.warning(f"[!] Skipped chat line: {line} ({e})")



    await db.commit()
    await run_batch_matchmaker(db)

    return {
        "status": "success",
        "total_lines_scanned": len(lines),
        "leads_extracted": len(parsed_leads),
        "leads": parsed_leads
    }