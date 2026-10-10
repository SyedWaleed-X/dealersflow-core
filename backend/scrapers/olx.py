import re
import asyncio
import logging
from typing import List, Dict, Any
from backend.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

PAK_PHONE_REGEX = re.compile(r"(?:(\+92|03)\d{2}[-\s]?\d{7})")


class OLXScraper(BaseScraper):
    """
    Pagination-aware OLX scraper: Crawls sequential pages (?page=1, ?page=2)
    until it fulfills the requested quota of listings.
    """

    async def scrape(self, target_url: str, max_items: int = 15) -> List[Dict[str, Any]]:
        print(f"[*] [OLX] Scraping feed with quota {max_items}: {target_url}...")
        extracted: List[Dict[str, Any]] = []
        page_num = 1
        max_pages = max(1, (max_items // 8) + 1)

        p, browser, page = None, None, None

        try:
            p, browser, page = await self.get_stealth_page()

            while len(extracted) < max_items and page_num <= max_pages:
                # Append page query parameter
                separator = "&" if "?" in target_url else "?"
                current_page_url = f"{target_url}{separator}page={page_num}" if page_num > 1 else target_url

                print(f"[*] [OLX] Crawling Page {page_num} ({current_page_url})...")

                response = await page.goto(
                    current_page_url,
                    timeout=self.NAVIGATION_TIMEOUT_MS,
                    wait_until="domcontentloaded"
                )

                if response and response.status in [403, 429, 503]:
                    print(f"[!] OLX WAF block ({response.status}). Tripping circuit breaker.")
                    await browser.close()
                    return self.load_fallback_data()

                await self.simulate_human_behavior(page)

                cards = await page.locator("article, [data-aut-id='itemBox'], li[data-aut-id]").all()
                if not cards:
                    print(f"[!] No more cards found on page {page_num}. Ending pagination.")
                    break

                for card in cards:
                    if len(extracted) >= max_items:
                        break

                    card_text = await card.inner_text()
                    lines = [l.strip() for l in card_text.split("\n") if l.strip()]
                    if len(lines) < 2:
                        continue

                    posted_date = "Recently"
                    date_elem = card.locator('[aria-label="Creation date"]').first
                    if await date_elem.count() > 0:
                        posted_date = (await date_elem.text_content() or "Recently").strip()
                    else:
                        for line in lines:
                            if any(t in line.lower() for t in ["ago", "yesterday"]):
                                posted_date = line.split("•")[-1].strip() if "•" in line else line
                                break

                    price_str = ""
                    for line in lines:
                        if "rs" in line.lower() or "pkr" in line.lower() or any(c in line.lower() for c in ["crore", "lac", "lakh"]):
                            price_str = line
                            break

                    title_str = lines[1] if len(lines) > 1 and lines[0] == price_str else lines[0]
                    if title_str.lower() in ["featured", "verified"] and len(lines) >= 3:
                        title_str = lines[2] if lines[1] == price_str else lines[1]

                    location_str = "Pakistan"
                    loc_elem = card.locator('[aria-label="Location"]').first
                    if await loc_elem.count() > 0:
                        location_str = (await loc_elem.text_content() or "Pakistan").strip()
                    else:
                        for line in lines:
                            if "•" in line and any(t in line.lower() for t in ["ago", "day", "hour", "week"]):
                                location_str = line.split("•")[0].strip()
                                break

                    link_elem = card.locator("a").first
                    href = await link_elem.get_attribute("href") or ""
                    full_url = f"https://www.olx.com.pk{href}" if href.startswith("/") else href

                    img_elem = card.locator("img").first
                    img_src = await img_elem.get_attribute("src") or await img_elem.get_attribute("data-src") or ""

                    phone_match = PAK_PHONE_REGEX.search(card_text)
                    contact_phone = phone_match.group(0) if phone_match else None

                    extracted.append({
                        "title": title_str,
                        "raw_price": price_str or "Demand on Call",
                        "location": location_str,
                        "posted_date": posted_date,
                        "listing_url": full_url,
                        "image_url": img_src,
                        "contact_phone": contact_phone,
                        "raw_transcript": card_text,
                        "source": "olx_pakistan"
                    })

                page_num += 1
                await asyncio.sleep(1.5)

            await browser.close()
            print(f"[+] [OLX] Successfully collected {len(extracted)} listings across {page_num-1} pages!")
            return extracted

        except Exception as e:
            print(f"[!] [OLX] Exception ({type(e).__name__}: {e}). Tripping circuit breaker.")
            if browser:
                await browser.close()
            return self.load_fallback_data()