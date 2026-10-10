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
Analyze colloquial Roman-Urdu, Urdu Nastaliq script, and English transcripts and extract structured property lead details into valid JSON.

REGIONAL PHONETIC & SCRIPT EQUIVALENCES (HANDLE TYPOS):
- CITIES:
  * "لہور", "لاہور", "lahore", "lhr" -> City: "Lahore"
  * "اسلام آباد", "اسلامہ باد", "islamabad", "isb" -> City: "Islamabad"
  * "گوجرانوالہ", "gujranwala", "grw" -> City: "Gujranwala"
  * "کراچی", "karachi", "khi" -> City: "Karachi"
  * "راولپنڈی", "پنڈی", "rawalpindi", "pindi" -> City: "Rawalpindi"
  * "ملتان", "multan" -> City: "Multan"
- NUMBER WORDS (CRITICAL):
  * "ساتھ", "سات", "saat", "sat" -> 7
  * "چھ", "chhe" -> 6
  * "پانچ", "paanch" -> 5
  * "چار", "char" -> 4
  * "تین", "teen" -> 3
  * "دو", "do" -> 2
  * "ایک", "ek", "aik" -> 1
  * "دس", "das" -> 10
  * "بیس", "bees" -> 20
  * "پچاس", "pachaas" -> 50
- CURRENCY & AREA UNITS:
  * "کروڈ", "پڑھوڑ", "کروڑ", "karor", "crore", "cr" -> crore
  * "لاکھ", "لیکھ", "lac", "lakh" -> lac
  * "مرلے", "مرلہ", "مدلے", "marla" -> marla
  * "کنال", "kanal" -> kanal
- INTENT:
  * "چاہیے", "خریدنا", "chahiye", "khareedna", "budget", "need", "require" -> BUY
  * "بیچنا", "bechna", "for sale", "demand", "available" -> SELL

Output ONLY valid JSON matching this schema:
{
  "intent": "BUY" | "SELL",
  "property_type": "PLOT" | "HOUSE" | "COMMERCIAL",
  "city": "Lahore" | "Islamabad" | "Gujranwala" | "Karachi" | "Rawalpindi" | "Multan" | null,
  "society": "string or null",
  "raw_price": "exact price phrase (e.g. '3 crore', '80 lac') or null",
  "raw_area": "exact area phrase (e.g. '7 marla', '1 kanal') or null",
  "client_name": "string or null",
  "client_phone": "string or null"
}
"""

BATCH_CHAT_PROMPT = """
You are an expert Real Estate Intelligence Dissector.
You are given a raw copy-pasted WhatsApp group chat log containing messages from multiple people.
1. Ignore personal small talk, greetings, banter, and jokes (e.g. 'kab tak ayoge', 'shuttle le ana').
2. Identify messages where someone is buying, selling, or asking about real estate.
3. For EVERY valid real estate mention, extract a distinct property lead, tagging the sender's name from the chat log.

Return a JSON array of leads matching the schema. If no leads exist, return [].
"""


class LeadExtractor:
    """
    Hybrid Lead Extraction Engine:
    Handles Nastaliq Urdu script, Roman-Urdu, English, and phonetic misspellings natively.
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
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty.")

        if self.intake_provider == LLMProvider.GEMINI and self.gemini_key:
            return self._extract_via_gemini(transcript)
        else:
            return self._extract_via_ollama(transcript, max_retries)

    def extract_batch_chat(self, chat_log: str) -> List[dict]:
        if not chat_log or not chat_log.strip():
            return []

        if self.batch_provider == LLMProvider.GEMINI and self.gemini_key:
            return self._extract_batch_via_gemini(chat_log)
        else:
            return self._extract_batch_via_ollama(chat_log)

    # GOOGLE AI STUDIO (GEMINI 3.8 FLASH)
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

    # LOCAL OLLAMA (QWEN 3.5:9B)
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
                response = httpx.post(self.ollama_api_url, json=payload, timeout=60.0)
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