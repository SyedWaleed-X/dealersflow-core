import abc
import random
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from playwright.async_api import async_playwright, Browser, Page

try:
    from playwright_stealth import Stealth
except ImportError:
    Stealth = None

logger = logging.getLogger(__name__)


class BaseScraper(abc.ABC):
    """
    Abstract Base Scraper with human behavior emulation, 
    smooth micro-scrolling, jitter delays, and failsafe fallback.
    """

    DEFAULT_FALLBACK_PATH = Path("data/pakistan_real_estate_fallback.csv")
    NAVIGATION_TIMEOUT_MS = 25000

    def __init__(self, fallback_csv_path: Optional[str | Path] = None):
        self.fallback_csv_path = Path(fallback_csv_path) if fallback_csv_path else self.DEFAULT_FALLBACK_PATH

    @abc.abstractmethod
    async def scrape(self, target_url: str, max_items: int = 10) -> List[Dict[str, Any]]:
        pass

    async def get_stealth_page(self) -> tuple[Any, Browser, Page]:
        pw_context = Stealth().use_async(async_playwright()) if Stealth else async_playwright()
        p = await pw_context.__aenter__()

        browser: Browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars"
            ]
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US"
        )

        page: Page = await context.new_page()
        return p, browser, page

    async def simulate_human_behavior(self, page: Page) -> None:
        """
        Emulates natural human reading: gentle mouse jitter and realistic micro-scrolling.
        Bypasses modern Cloudflare and DataDome behavioral telemetry checks.
        """
        # 1. Random natural pause (reading headline)
        await asyncio.sleep(random.uniform(1.2, 2.2))

        # 2. Smooth downward scroll in random increments
        for _ in range(3):
            scroll_delta = random.randint(250, 450)
            await page.evaluate(f"window.scrollBy(0, {scroll_delta});")
            await asyncio.sleep(random.uniform(0.3, 0.7))

        # 3. Micro mouse jitter
        await page.mouse.move(random.randint(100, 700), random.randint(200, 600))
        await asyncio.sleep(random.uniform(0.5, 1.0))

    def load_fallback_data(self) -> List[Dict[str, Any]]:
        if not self.fallback_csv_path.exists():
            raise FileNotFoundError(f"Fallback CSV missing at {self.fallback_csv_path}")

        print(f"\n[CIRCUIT BREAKER ACTIVATED] Ingesting verified local archive from: {self.fallback_csv_path}")
        df = pd.read_csv(self.fallback_csv_path)

        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            records.append({
                "title": str(row["title"]),
                "city": str(row["city"]),
                "society": str(row["society"]),
                "phase": str(row["phase"]),
                "raw_price": str(row["raw_price"]),
                "raw_area": str(row["raw_area"]),
                "property_type": str(row["property_type"]),
                "intent": str(row["intent"]),
                "listing_url": "https://www.olx.com.pk/properties_c2",
                "image_url": "",
                "posted_date": "Recently",
                "contact_phone": str(row["contact_phone"]),
                "source": "offline_fallback"
            })

        return records