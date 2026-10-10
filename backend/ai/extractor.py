import time
import json
import logging
from typing import List, Optional
import httpx

from backend.core.config import settings, LLMProvider
from backend.domain.schemas import ExtractedPropertyLead

logger = logging.getLogger(__name__)

SYSTEM_EXTRACTION_PROMPT = """
You are an expert South Asian Real Estate Lead Extraction Engine.
Analyze colloquial Roman-Urdu, Urdu Nastaliq, and English voice note transcripts and extract structured property lead details into valid JSON.

REGIONAL CONVENTIONS:
- INTENT:
  * "bechna", "for sale", "demand", "available", "meri offer" -> SELL
  * "khareedna", "chahiye", "require", "budget", "need", "talash" -> BUY
- PROPERTY TYPES:
  * "plot", "file", "parchi", "qist" -> PLOT
  * "kothi", "makan", "ghar", "bangla", "flat", "apartment" -> HOUSE
  * "plaza", "dukan", "shop", "office", "hall", "commercial" -> COMMERCIAL
- MEASUREMENT & PRICING:
  * Keep raw pricing expressions intact (e.g., '4 crore', '85 lac', '1.5 cr').
  * Keep raw area expressions intact (e.g., '1 kanal', '10 marla', '500 gaz').
- CLIENT & PHONE:
  * Extract personal names and Pakistani phone numbers.

Output ONLY valid JSON matching the schema directly. Zero conversational preambles.
"""

BATCH_CHAT_PROMPT = """
You are an expert Real Estate Intelligence Dissector.
You are given a raw copy-pasted WhatsApp group chat log containing messages from multiple people.
Your job is to:
1. Ignore all personal small talk, greetings, banter, and jokes (e.g. 'kab tak ayoge', 'shuttle le ana', 'match kon jeeta').
2. Identify messages where someone is buying, selling, or asking about real estate (plots, houses, shops, files).
3. For EVERY valid real estate mention, extract a distinct property lead, tagging the sender's name from the chat log.

Return a JSON array of leads matching this schema:
[
  {
    "intent": "BUY" | "SELL",
    "property_type": "PLOT" | "HOUSE" | "COMMERCIAL",
    "city": "string",
    "society": "string",
    "raw_price": "string or null",
    "raw_area": "string or null",
    "client_name": "Sender Name from chat log",
    "client_phone": "Phone number if present or null"
  }
]
If no real estate leads exist, return an empty array []. Zero conversational text.
"""


class LeadExtractor:
    """
    Hybrid Lead Extraction Engine:
    - INTAKE_LLM_PROVIDER: Governs single voice notes & text leads.
    - BATCH_LLM_PROVIDER: Governs 100-message bulk WhatsApp group chat logs.
    """

    def __init__(self):
        self.intake_provider = settings.INTAKE_LLM_PROVIDER
        self.batch_provider = settings.BATCH_LLM_PROVIDER
        self.gemini_key = settings.GEMINI_API_KEY
        self.gemini_model = settings.GEMINI_MODEL

        raw_url = settings.active_llm_base_url.replace("/v1", "").rstrip("/")
        self.ollama_api_url = f"{raw_url}/api/chat"
        self.ollama_model = settings.LOCAL_LLM_MODEL

    def extract(self, transcript: str, max_retries: int = 2) -> ExtractedPropertyLead:
        """Extracts a single lead according to INTAKE_LLM_PROVIDER setting."""
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty.")

        if self.intake_provider == LLMProvider.GEMINI and self.gemini_key:
            return self._extract_via_gemini(transcript)
        else:
            return self._extract_via_ollama(transcript, max_retries)

    def extract_batch_chat(self, chat_log: str) -> List[dict]:
        """Dissects bulk WhatsApp chat history according to BATCH_LLM_PROVIDER setting."""
        if not chat_log or not chat_log.strip():
            return []

        if self.batch_provider == LLMProvider.GEMINI and self.gemini_key:
            return self._extract_batch_via_gemini(chat_log)
        else:
            return self._extract_batch_via_ollama(chat_log)

    # ==========================================
    # GOOGLE AI STUDIO (GEMINI 3.8 FLASH)
    # ==========================================
    def _extract_via_gemini(self, transcript: str) -> ExtractedPropertyLead:
        print(f"\n[*] [GOOGLE AI STUDIO] Extracting single lead via {self.gemini_model}...")
        t0 = time.perf_counter()

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_key)

        response = client.models.generate_content(
            model=self.gemini_model,
            contents=f"{SYSTEM_EXTRACTION_PROMPT}\n\nTranscript:\n\"\"\"{transcript}\"\"\"",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedPropertyLead,
                temperature=0.0
            )
        )

        elapsed = time.perf_counter() - t0
        print(f"[+] [GOOGLE AI STUDIO] Completed in {elapsed:.2f} seconds!")
        return ExtractedPropertyLead.model_validate_json(response.text)

    def _extract_batch_via_gemini(self, chat_log: str) -> List[dict]:
        print(f"\n[*] [GOOGLE AI STUDIO] Dissecting WhatsApp chat log via {self.gemini_model}...")
        t0 = time.perf_counter()

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_key)

        # Direct schema generation with tools explicitly empty to kill the 100s AFC loop!
        response = client.models.generate_content(
            model=self.gemini_model,
            contents=f"{BATCH_CHAT_PROMPT}\n\nChat Log:\n\"\"\"{chat_log}\"\"\"",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ExtractedPropertyLead],
                temperature=0.0
            )
        )

        elapsed = time.perf_counter() - t0
        print(f"[+] [GOOGLE AI STUDIO] Bulk chat parsed in {elapsed:.2f} seconds!")
        raw_items = json.loads(response.text)
        return raw_items if isinstance(raw_items, list) else [raw_items]

    # ==========================================
    # LOCAL OLLAMA (RTX 3060 QWEN3.5:9B)
    # ==========================================
    def _extract_via_ollama(self, transcript: str, max_retries: int) -> ExtractedPropertyLead:
        print(f"\n[*] [GPU TENSOR COMPUTE] Qwen3.5-9B active on RTX 3060...")
        t0 = time.perf_counter()

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Transcript:\n\"\"\"{transcript}\"\"\""}
            ],
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": 8192},
            "stream": False
        }

        for attempt in range(max_retries + 1):
            try:
                response = httpx.post(self.ollama_api_url, json=payload, timeout=120.0)
                response.raise_for_status()

                content = response.json()["message"]["content"]
                lead = ExtractedPropertyLead.model_validate_json(content)

                elapsed = time.perf_counter() - t0
                print(f"[+] [GPU TENSOR COMPUTE] Qwen3.5-9B completed in {elapsed:.2f} seconds!")
                return lead
            except Exception as e:
                logger.warning(f"[!] Extraction attempt {attempt+1} failed ({e}). Retrying...")

        raise ValueError("Local Ollama extraction failed after retries.")

    def _extract_batch_via_ollama(self, chat_log: str) -> List[dict]:
        print(f"\n[*] [GPU TENSOR COMPUTE] Parsing chat history locally via Qwen3.5-9B...")
        t0 = time.perf_counter()

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": BATCH_CHAT_PROMPT},
                {"role": "user", "content": f"Chat Log:\n\"\"\"{chat_log}\"\"\""}
            ],
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": 16384},
            "stream": False
        }

        response = httpx.post(self.ollama_api_url, json=payload, timeout=90.0)
        response.raise_for_status()
        content = response.json()["message"]["content"]
        raw_items = json.loads(content)

        elapsed = time.perf_counter() - t0
        print(f"[+] [GPU TENSOR COMPUTE] Local batch chat parsed in {elapsed:.2f} seconds!")
        return raw_items if isinstance(raw_items, list) else [raw_items]