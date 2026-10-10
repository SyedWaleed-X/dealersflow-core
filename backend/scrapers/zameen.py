import re
import asyncio
import logging
from typing import List, Dict, Any
from backend.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

ZAMEEN_DATE_REGEX = re.compile(r"Added:\s*([^\(\n•]+)", re.IGNORECASE)


class ZameenScraper(BaseScraper):
    """
    Pagination-aware Zameen.com scraper: Crawls sequential pages
    (e.g. Lahore-1-1.html -> Lahore-1-2.html) until fulfilling quota.
    """

    async def scrape(self, target_url: str, max_items: int = 15) -> List[Dict[str, Any]]:
        print(f"[*] [ZAMEEN] Scraping feed with quota {max_items}: {target_url}...")
        extracted: List[Dict[str, Any]] = []
        page_num = 1
        max_pages = max(1, (max_items // 8) + 1)

        p, browser, page = None, None, None

        try:
            p, browser, page = await self.get_stealth_page()

            while len(extracted) < max_items and page_num <= max_pages:
                # Zameen URL pagination format: Lahore-1-1.html -> Lahore-1-2.html
                if page_num > 1 and "-1.html" in target_url:
                    current_page_url = target_url.replace("-1.html", f"-{page_num}.html")
                else:
                    current_page_url = target_url

                print(f"[*] [ZAMEEN] Crawling Page {page_num} ({current_page_url})...")

                response = await page.goto(
                    current_page_url,
                    timeout=self.NAVIGATION_TIMEOUT_MS,
                    wait_until="domcontentloaded"
                )

                if response and response.status in [403, 429, 503]:
                    print(f"[!] Zameen WAF block ({response.status}). Tripping circuit breaker.")
                    await browser.close()
                    return self.load_fallback_data()

                await self.simulate_human_behavior(page)

                cards = await page.locator("article, [aria-label='Listing'], li[role='article']").all()
                if not cards:
                    print(f"[!] No more Zameen cards on page {page_num}. Ending pagination.")
                    break

                for card in cards:
                    if len(extracted) >= max_items:
                        break

                    card_text = await card.inner_text()
                    lines = [l.strip() for l in card_text.split("\n") if l.strip()]
                    if len(lines) < 2:
                        continue

                    date_match = ZAMEEN_DATE_REGEX.search(card_text)
                    posted_date = date_match.group(1).strip() if date_match else "Recently"

                    price_str = ""
                    for line in lines:
                        if "pkr" in line.lower() or any(c in line.lower() for c in ["crore", "lakh", "lac"]):
                            price_str = line
                            break

                    title_str = "Zameen Property Listing"
                    for line in lines:
                        if line != price_str and len(line) > 12 and not line.lower().startswith("added:"):
                            title_str = line
                            break

                    location_str = lines[-1] if len(lines) >= 3 else "Pakistan"

                    link_elem = card.locator("a").first
                    href = await link_elem.get_attribute("href") or ""
                    full_url = f"https://www.zameen.com{href}" if href.startswith("/") else href

                    img_elem = card.locator("img").first
                    img_src = await img_elem.get_attribute("src") or await img_elem.get_attribute("data-src") or ""

                    extracted.append({
                        "title": title_str,
                        "raw_price": price_str or "Demand on Call",
                        "location": location_str,
                        "posted_date": posted_date,
                        "listing_url": full_url,
                        "image_url": img_src,
                        "contact_phone": None,
                        "raw_transcript": card_text,
                        "source": "zameen_com"
                    })

                page_num += 1
                await asyncio.sleep(1.5)

            await browser.close()
            print(f"[+] [ZAMEEN] Successfully collected {len(extracted)} listings across {page_num-1} pages!")
            return extracted

        except Exception as e:
            print(f"[!] [ZAMEEN] Exception ({type(e).__name__}: {e}). Tripping circuit breaker.")
            if browser:
                await browser.close()
            return self.load_fallback_data()