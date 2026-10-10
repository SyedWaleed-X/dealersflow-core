import logging
import urllib.parse
from typing import Dict, Any
from backend.ai.graph_state import AgentPipelineState
from backend.ai.asr_engine import Transcriber
from backend.ai.audio_processor import AudioProcessor
from backend.ai.extractor import LeadExtractor
from backend.domain.schemas import NormalizedPropertyLead
from backend.ai.valuation import valuation_service
from backend.ai.matcher import LeadMatcher

logger = logging.getLogger(__name__)

# Singletons for graph execution
transcriber = Transcriber()
extractor = LeadExtractor()


# ==========================================
# NODE 1: AUDIO INGESTION & ASR
# ==========================================
def transcribe_node(state: AgentPipelineState) -> Dict[str, Any]:
    """Transcribes audio file to Latin-script Roman-Urdu, or passes raw text directly."""
    print("\n--- [NODE 1] Audio Ingestion & Transcription ---")

    if state.get("raw_text"):
        transcript = state["raw_text"].strip()
        print(f"[*] Raw text input received: \"{transcript[:50]}...\"")
        return {"transcript": transcript, "error": None}

    audio_path = state.get("audio_path")
    if not audio_path:
        return {"transcript": "", "error": "No audio path or raw text provided.", "is_success": False}

    try:
        # Convert to 16kHz mono WAV using Day 3 processor
        wav_path = AudioProcessor.convert_to_wav_16k(audio_path)
        transcript = transcriber.transcribe(wav_path)
        AudioProcessor.cleanup_temp_file(wav_path)
        print(f"[+] Roman-Urdu Transcript: \"{transcript}\"")
        return {"transcript": transcript, "error": None}
    except Exception as e:
        logger.error(f"Transcribe node error: {e}")
        return {"transcript": "", "error": f"ASR Error: {str(e)}", "is_success": False}


# ==========================================
# NODE 2: STRUCTURED EXTRACTION & NORMALIZATION
# ==========================================
def extract_node(state: AgentPipelineState) -> Dict[str, Any]:
    """Extracts Pydantic entities via Qwen3.5 and normalizes South Asian currency/land units."""
    print("--- [NODE 2] Structured Entity Extraction & Normalization ---")
    transcript = state.get("transcript", "").strip()

    if not transcript:
        return {"error": "Empty transcript cannot be extracted.", "is_success": False}

    try:
        # Extract via Qwen3.5 + Instructor
        extracted_pydantic = extractor.extract(transcript)
        # Normalize units via Day 5 normalizer
        normalized_pydantic = NormalizedPropertyLead.from_extracted_lead(extracted_pydantic)

        print(f"[+] Intent: {normalized_pydantic.intent} | City: {normalized_pydantic.city} | Society: {normalized_pydantic.society}")
        print(f"[+] Normalized Price: {normalized_pydantic.price_pkr:,} PKR | Area: {normalized_pydantic.area_sqft} sqft")

        return {
            "extracted_lead": extracted_pydantic.model_dump(),
            "normalized_lead": normalized_pydantic.model_dump(),
            "error": None
        }
    except Exception as e:
        logger.error(f"Extract node error: {e}")
        return {"error": f"Extraction Error: {str(e)}", "is_success": False}


# ==========================================
# NODE 3: VALUATION & HYBRID MATCHMAKING
# ==========================================
async def valuation_match_node(state: AgentPipelineState) -> Dict[str, Any]:
    """Computes CatBoost fair-market price and queries counter-leads in PostgreSQL."""
    print("--- [NODE 3] Valuation & Hybrid Matchmaking ---")
    norm_lead = state.get("normalized_lead")

    if not norm_lead or not norm_lead.get("price_pkr"):
        return {"valuation_result": None, "matched_leads": [], "error": "Missing normalized price for valuation."}

    city = norm_lead["city"]
    society = norm_lead["society"]
    phase = norm_lead.get("phase") or "General"
    prop_type = norm_lead["property_type"]
    area_sqft = norm_lead["area_sqft"] or 2250.0
    asking_price = norm_lead["price_pkr"]
    intent = norm_lead["intent"]

    # 1. CatBoost Valuation & Prediction Interval
    val_pred = valuation_service.predict_fair_value(
        city=city,
        society=society,
        phase=phase,
        property_type=prop_type,
        area_sqft=area_sqft,
        bedrooms=norm_lead.get("bedrooms", 0),
        bathrooms=norm_lead.get("bathrooms", 0)
    )

    # 2. Arbitrage Deal Rating
    deal_eval = valuation_service.evaluate_market_deal(
        asking_price_pkr=asking_price,
        estimated_price_pkr=val_pred["estimated_price_pkr"],
        lower_bound_pkr=val_pred["lower_bound_pkr"],
        upper_bound_pkr=val_pred["upper_bound_pkr"]
    )

    valuation_payload = {**val_pred, **deal_eval}
    print(f"[+] Fair Value: {val_pred['estimated_price_pkr']:,} PKR | Status: {deal_eval['deal_status']}")

    # 3. Two-Stage Hybrid Vector Match
    matched = await LeadMatcher.match_lead(
        intent=intent,
        city=city,
        target_budget_pkr=asking_price,
        tolerance_percentage=0.25,
        preference_text=f"{society} {' '.join(norm_lead.get('qualitative_features', []))}",
        limit=3
    )
    print(f"[+] Found {len(matched)} matching counter-party leads in {city}.")

    return {
        "valuation_result": valuation_payload,
        "matched_leads": matched,
        "error": None
    }


# ==========================================
# NODE 4: OUTBOUND SYNTHESIS & WHATSAPP LINK
# ==========================================
def response_node(state: AgentPipelineState) -> Dict[str, Any]:
    """Drafts professional negotiation outreach and encodes RFC-3986 wa.me deep link."""
    print("--- [NODE 4] Response & WhatsApp Synthesis ---")
    norm_lead = state.get("normalized_lead")
    val_res = state.get("valuation_result", {})
    matched = state.get("matched_leads", [])

    client_name = norm_lead.get("client_name") or "Sir/Madam"
    client_phone = norm_lead.get("client_phone") or "923001234567"
    # Normalize phone digits (e.g. 03001234567 -> 923001234567)
    phone_digits = "".join(filter(str.isdigit, client_phone))
    if phone_digits.startswith("0"):
        phone_digits = "92" + phone_digits[1:]

    # Construct negotiation follow-up message
    msg = (
        f"Aoa {client_name},\n\n"
        f"Regarding your {norm_lead['property_type']} in {norm_lead['society']}, {norm_lead['city']}:\n"
        f"• Demand: {norm_lead['price_pkr']:,} PKR\n"
        f"• AI Fair-Market Valuation: {val_res.get('estimated_price_pkr', 0):,} PKR\n"
        f"• Deal Rating: {val_res.get('deal_status', 'FAIR_MARKET')}\n\n"
        f"We currently have {len(matched)} active counter-leads in our network ready for viewing.\n"
        f"Let me know if you would like to arrange a meeting."
    )

    # Encode RFC-3986 wa.me deep link
    encoded_text = urllib.parse.quote(msg)
    wa_link = f"https://wa.me/{phone_digits}?text={encoded_text}"

    print(f"[+] Generated wa.me Deep Link: {wa_link[:60]}...")

    return {
        "whatsapp_message": msg,
        "whatsapp_deep_link": wa_link,
        "is_success": True
    }