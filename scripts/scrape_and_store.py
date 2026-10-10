import asyncio
import json
import random
import warnings
from sqlalchemy import text
from backend.scrapers.olx import OLXScraper
from backend.scrapers.zameen import ZameenScraper
from backend.normalizer import PropertyNormalizer
from backend.database import AsyncSessionLocal, engine

# Suppress harmless Windows pipe teardown warnings on script exit
warnings.filterwarnings("ignore", category=ResourceWarning)

# Verified Portals
OLX_PORTALS = [
    {"city": "Gujranwala", "url": "https://www.olx.com.pk/gujranwala_g4060662/properties_c2"},
    {"city": "Lahore", "url": "https://www.olx.com.pk/lahore_g4060673/properties_c2"},
    {"city": "Islamabad", "url": "https://www.olx.com.pk/islamabad_g4060615/properties_c2"},
    {"city": "Karachi", "url": "https://www.olx.com.pk/karachi_g4060695/properties_c2"},
]

ZAMEEN_PORTALS = [
    {"city": "Lahore", "url": "https://www.zameen.com/Homes/Lahore-1-1.html"},
    {"city": "Islamabad", "url": "https://www.zameen.com/Homes/Islamabad-3-1.html"},
]

PAK_CITIES = ["Gujranwala", "Lahore", "Islamabad", "Karachi", "Rawalpindi", "Faisalabad", "Multan"]

KNOWN_SOCIETIES = [
    "master city", "ajwa city", "citi housing", "dc colony", "royal palm city", "wapda city",
    "dha phase 6", "dha phase 5", "dha phase 8", "dha phase 2", "dha sector g", "dha",
    "bahria town", "bahria orchard", "wapda town", "johar town", "gulberg", "margalla enclave",
    "soan garden", "g-11", "f-11", "f-7", "askari 14", "clifton", "gulshan"
]


def clean_location(loc_text: str, title: str, default_city: str) -> tuple[str, str]:
    text_corpus = (loc_text + " " + title).lower().replace("-", " ")
    city = default_city
    for c in PAK_CITIES:
        if c.lower() in text_corpus:
            city = c
            break

    society = "Central Zone"
    for s in KNOWN_SOCIETIES:
        if s in text_corpus:
            society = s.title()
            break
    if society == "Central Zone" and "," in loc_text:
        parts = [p.strip() for p in loc_text.split(",") if p.strip()]
        if len(parts) >= 2:
            society = parts[0]

    return city, society


async def main():
    print("=" * 80)
    print("MODULAR MULTI-SOURCE HARVESTER (OLX PAKISTAN + ZAMEEN.COM)")
    print("=" * 80)

    olx_scraper = OLXScraper()
    zameen_scraper = ZameenScraper()
    all_leads = []

    # 1. Harvest OLX
    for portal in OLX_PORTALS:
        city = portal["city"]
        url = portal["url"]
        listings = await olx_scraper.scrape(url, max_items=5)
        for item in listings:
            item["target_city"] = city
            all_leads.append(item)
        
        jitter = random.uniform(2.5, 4.0)
        print(f"[*] Human pause: waiting {jitter:.1f}s...")
        await asyncio.sleep(jitter)

    # 2. Harvest Zameen.com
    for portal in ZAMEEN_PORTALS:
        city = portal["city"]
        url = portal["url"]
        listings = await zameen_scraper.scrape(url, max_items=5)
        for item in listings:
            item["target_city"] = city
            all_leads.append(item)

        jitter = random.uniform(2.5, 4.0)
        print(f"[*] Human pause: waiting {jitter:.1f}s...")
        await asyncio.sleep(jitter)

    print(f"\n[+] Total Live Listings Captured: {len(all_leads)}")
    print("[*] Ingesting into PostgreSQL (with ON CONFLICT deduplication)...")

    saved_count = 0
    duplicate_count = 0

    async with AsyncSessionLocal() as session:
        for item in all_leads:
            title = item.get("title", "Property Listing")
            raw_price = item.get("raw_price", "")
            target_city = item.get("target_city", "Lahore")
            location_str = item.get("location", target_city)
            posted_date = item.get("posted_date", "Recently")
            listing_url = item.get("listing_url", "")
            image_url = item.get("image_url", "")
            source = item.get("source", "web_scraper")
            contact_phone = item.get("contact_phone")
            transcript = item.get("raw_transcript", title)

            city, society = clean_location(location_str, title, target_city)
            price_pkr = PropertyNormalizer.normalize_price(raw_price) or 15_000_000
            area_sqft = PropertyNormalizer.normalize_area(transcript) or 1125.0

            prop_type = "PLOT"
            lower_text = (title + " " + transcript).lower()
            if any(k in lower_text for k in ["house", "makan", "kothi", "portion", "flat", "apartment", "villa"]):
                prop_type = "HOUSE"
            elif any(k in lower_text for k in ["commercial", "shop", "plaza", "office"]):
                prop_type = "COMMERCIAL"

            features_dict = {
                "image_url": image_url,
                "posted_date": posted_date,
                "scraped_title": title,
                "data_source": source
            }

            # Clean, native PostgreSQL unique column conflict handling
            insert_stmt = text("""
                INSERT INTO properties (
                    intent, prop_type, city, society, price_pkr, area_sqft, 
                    raw_transcript, features, contact_phone, listing_url
                ) VALUES (
                    'SELL'::property_intent,
                    CAST(:prop_type AS property_type),
                    :city,
                    :society,
                    :price_pkr,
                    :area_sqft,
                    :raw_transcript,
                    CAST(:features AS jsonb),
                    :contact_phone,
                    :listing_url
                )
                ON CONFLICT (listing_url) DO NOTHING;
            """)

            res = await session.execute(insert_stmt, {
                "prop_type": prop_type,
                "city": city[:64],
                "society": society[:128],
                "price_pkr": price_pkr,
                "area_sqft": area_sqft,
                "raw_transcript": transcript,
                "features": json.dumps(features_dict),
                "contact_phone": contact_phone,
                "listing_url": listing_url
            })

            # res.rowcount tells us whether a new row was inserted or skipped as duplicate
            if res.rowcount > 0:
                saved_count += 1
                print(f"  [SAVED NEW] [{source.upper()}] {city.upper()} | {society} | {price_pkr:,} PKR | Date: {posted_date}")
            else:
                duplicate_count += 1
                print(f"  [SKIPPED DUP] [{source.upper()}] {city.upper()} | {title[:35]}...")

        await session.commit()
        count_res = await session.execute(text("SELECT COUNT(*) FROM properties;"))
        total_in_db = count_res.scalar()

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Ingested {saved_count} NEW properties ({duplicate_count} duplicates skipped)!")
    print(f"[+] Total records now in PostgreSQL 'properties' table: {total_in_db}")
    print("=" * 80)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())