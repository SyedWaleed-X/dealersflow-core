import re
import time
import json
import random
import asyncio
import logging
from typing import List, Dict, Any, Optional
from ddgs import DDGS
from sqlalchemy import text

from backend.core.database import AsyncSessionLocal
from backend.ai.embedder import embedding_engine
from backend.domain.normalizer import PropertyNormalizer

logger = logging.getLogger("dealersflow.hunter")

# Whitelist credible real estate platforms & social sources in Pakistan
REAL_ESTATE_DOMAINS = ["olx.com.pk", "zameen.com", "graana.com", "facebook.com", "instagram.com", "lamudi.pk", "property"]
JUNK_DOMAINS = ["youtube.com", "poki.com", "google.com", "tiktok.com", "wikipedia.org", "game"]


class AutonomousWebHunter:
    """
    Multi-Platform Autonomous Scouting Agent:
    Casts a wide net across Facebook, Graana, Lamudi, OLX, and Zameen.
    Applies human jitter delays (2.5s-4s) and preserves partial results on cancel.
    """

    MATCH_THRESHOLD = 0.65  # Relaxed slightly to capture rich social & open web posts
    CANCELLED = False

    @classmethod
    def request_cancel(cls):
        cls.CANCELLED = True
        print("[!] [WEB HUNTER] Cancellation flag set to TRUE. Halting search.")

    @classmethod
    def clean_search_term(cls, city: str, society: str, prop_type: str, raw_text: str = "") -> str:
        # Strip bullets (•), punctuation, and commas
        raw_soc = str(society or "").replace("•", " ").replace(",", " ").replace("-", " ")
        raw_soc = re.sub(r"\s+", " ", raw_soc).strip()

        clean_society = raw_soc if raw_soc.lower() not in ["general area", "general", "none", "pakistan"] else ""
        if not clean_society and raw_text:
            common_societies = [
                "dha phase 6", "dha phase 5", "dha phase 8", "dha phase 7", "dha phase 2", "dha",
                "bahria town phase 7", "bahria town phase 8", "bahria town", "bahria enclave", "bahria orchard",
                "citi housing", "dc colony", "royal palm city", "master city", "wapda city", "wapda town",
                "fazaia housing", "f-7", "f-8", "f-10", "f-11", "g-11", "g-13", "g-14", "johar town", "gulberg"
            ]
            for s in common_societies:
                if s in raw_text.lower():
                    clean_society = s.title()
                    break

        clean_city = str(city or "").replace("•", " ").replace(",", " ").strip()
        if clean_city.lower() in clean_society.lower():
            clean_society = re.sub(clean_city, "", clean_society, flags=re.IGNORECASE).strip()

        size_match = re.search(r"(\d+(?:\.\d+)?\s*(?:marla|kanal|gaz))", raw_text.lower())
        size_str = size_match.group(1) if size_match else ""

        parts = [clean_society, clean_city, size_str, prop_type.title()]
        query = " ".join([p for p in parts if p]).strip()
        query = re.sub(r"[^\w\s]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()

        return query or f"{clean_city} {prop_type.title()} for sale"

    @classmethod
    async def hunt_for_lead(
        cls,
        buyer_lead_id: Optional[str],
        city: str,
        society: str,
        property_type: str,
        budget_pkr: int,
        search_notes: str = "",
        duration_seconds: int = 60,
        max_results: int = 15
    ) -> Dict[str, Any]:
        cls.CANCELLED = False
        start_time = time.time()
        end_time = start_time + duration_seconds

        base_query = cls.clean_search_term(city, society, property_type, search_notes)
        print(f"\n[*] [WEB HUNTER] Multi-Platform Precision Query: \"{base_query}\"")

        # MULTI-SOURCE SEARCH EXPANSION: Open Web, Facebook, Portals, and Dealer Demand
        query_candidates = [
            f"{base_query} for sale",                      # Open Web (Graana, Lamudi, agency sites)
            f"{base_query} for sale facebook",             # Social media & public dealer posts
            f"{base_query} for sale olx zameen",           # Major classified portals
            f"{base_query} price demand",                  # Off-market pricing notes
        ]

        reference_text = f"{city} {society} {property_type} {base_query}".strip()
        buyer_vec = embedding_engine.embed_text(reference_text)

        inspected_audit_trail: List[Dict[str, Any]] = []
        discovered_matches: List[Dict[str, Any]] = []
        visited_urls = set()

        with DDGS() as ddgs:
            for query in query_candidates:
                if cls.CANCELLED or time.time() >= end_time or len(discovered_matches) >= max_results:
                    break

                print(f"[*] [WEB HUNTER] Querying web index: \"{query}\"...")
                try:
                    search_results = list(ddgs.text(query, max_results=8, region="pk-en"))
                except Exception as e:
                    logger.warning(f"[!] Query '{query}' failed: {e}")
                    continue

                for result in search_results:
                    if cls.CANCELLED or time.time() >= end_time:
                        break

                    url = result.get("href", "")
                    title = result.get("title", "")
                    snippet = result.get("body", "")

                    if not url or url in visited_urls:
                        continue
                    visited_urls.add(url)

                    # Spam filter: Drop gaming, streaming, and non-property noise
                    if any(junk in url.lower() for junk in JUNK_DOMAINS):
                        continue

                    # Vector similarity evaluation
                    result_text = f"{title} {snippet}"
                    result_vec = embedding_engine.embed_text(result_text)

                    similarity = sum(a * b for a, b in zip(buyer_vec, result_vec))
                    similarity = max(0.0, min(1.0, similarity))

                    is_match = similarity >= cls.MATCH_THRESHOLD
                    status_label = "MATCHED & SAVED" if is_match else "INSPECTED (LOW RELEVANCE)"

                    audit_entry = {
                        "url": url,
                        "title": title,
                        "snippet": snippet[:160] + "...",
                        "similarity_score": round(similarity, 4),
                        "status": status_label,
                        "is_match": is_match
                    }
                    inspected_audit_trail.append(audit_entry)

                    if is_match:
                        # Price sanitation: Prevents '5 PKR' glitch
                        parsed_price = PropertyNormalizer.normalize_price(snippet)
                        if not parsed_price or parsed_price < 50000:
                            parsed_price = budget_pkr  # Fallback to buyer's budget if price is unstated or extracted as 5

                        saved_id = await cls._persist_scout_result(
                            buyer_lead_id=buyer_lead_id,
                            title=title,
                            url=url,
                            snippet=snippet,
                            similarity=similarity,
                            price_pkr=parsed_price
                        )

                        discovered_matches.append({
                            "id": saved_id,
                            "title": title,
                            "url": url,
                            "price_pkr": parsed_price,
                            "similarity_score": round(similarity, 4)
                        })
                        print(f"  [+] [REAL DEAL MATCHED] {title[:45]}... ({(similarity*100):.1f}%)")

                # Anti-bot human delay: 2.5s to 3.5s
                if not cls.CANCELLED:
                    jitter = random.uniform(2.5, 3.5)
                    await asyncio.sleep(jitter)

        total_elapsed = time.time() - start_time
        print(f"[+] [WEB HUNTER] Hunt finished in {total_elapsed:.1f}s. Inspected {len(inspected_audit_trail)} links, found {len(discovered_matches)} deals!")

        return {
            "elapsed_seconds": round(total_elapsed, 1),
            "was_cancelled": cls.CANCELLED,
            "total_links_inspected": len(inspected_audit_trail),
            "matches_discovered_count": len(discovered_matches),
            "matches": discovered_matches,
            "audit_trail": sorted(inspected_audit_trail, key=lambda x: x["similarity_score"], reverse=True)
        }

    @classmethod
    async def _persist_scout_result(
        cls,
        buyer_lead_id: Optional[str],
        title: str,
        url: str,
        snippet: str,
        similarity: float,
        price_pkr: int
    ) -> Optional[str]:
        async with AsyncSessionLocal() as session:
            try:
                stmt = text("""
                    INSERT INTO web_scout_results (
                        buyer_id, title, url, snippet, similarity_score, price_pkr
                    ) VALUES (
                        CAST(:b_id AS uuid), :title, :url, :snippet, :score, :price
                    )
                    ON CONFLICT (buyer_id, url) DO UPDATE 
                    SET similarity_score = EXCLUDED.similarity_score,
                        price_pkr = EXCLUDED.price_pkr
                    RETURNING id;
                """)
                res = await session.execute(stmt, {
                    "b_id": buyer_lead_id,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "score": round(similarity, 4),
                    "price": price_pkr
                })
                await session.commit()
                row = res.fetchone()
                return str(row[0]) if row else None
            except Exception as e:
                logger.error(f"[!] Error saving scout result: {e}")
                return None