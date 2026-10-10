import os
import re
import urllib.parse
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class WhatsAppService:
    """
    Zero-Cost WhatsApp Orchestration Service:
    1. Sanitizes international and Pakistani phone numbers to MSISDN standard.
    2. Dynamically crafts negotiation messages with CatBoost valuations.
    3. Formats RFC-3986 compliant wa.me deep links with URI length safety caps.
    4. Provides optional Twilio Sandbox dispatch fallback.
    """

    MAX_URI_CHAR_LIMIT = 1500  # Mobile browsers truncate URLs beyond 2,000 characters

    @classmethod
    def sanitize_phone_number(cls, raw_phone: Optional[str]) -> str:
        """
        Normalizes phone numbers to WhatsApp's international format (MSISDN).
        Strips '+', dashes, spaces, and leading zeroes.
        Defaults Pakistani local format (03XX -> 923XX).
        """
        if not raw_phone or not str(raw_phone).strip():
            return "923001234567"  # Fallback default dealer number

        # Strip everything except digits
        digits = re.sub(r"\D", "", str(raw_phone))

        # Handle Pakistani local numbers: 03001234567 -> 923001234567
        if digits.startswith("0") and len(digits) == 11:
            digits = "92" + digits[1:]
        elif digits.startswith("92") and len(digits) == 12:
            pass  # Already proper format
        elif len(digits) == 10:
            # 3001234567 -> 923001234567
            digits = "92" + digits

        return digits

    @classmethod
    def generate_negotiation_template(
        cls,
        client_name: Optional[str],
        property_type: str,
        society: str,
        city: str,
        asking_price_pkr: int,
        valuation_dict: Optional[Dict[str, Any]] = None,
        matched_leads: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Drafts a high-converting negotiation follow-up referencing
        market valuations, price spreads, and matched active counter-leads.
        """
        name = client_name or "Respected Client"
        val_data = valuation_dict or {}
        matches = matched_leads or []

        estimated_val = val_data.get("estimated_price_pkr", 0)
        deal_status = val_data.get("deal_status", "FAIR_MARKET")
        spread_pct = val_data.get("spread_percentage", 0.0)

        # Strategic contextual advice tailored to the valuation spread
        if deal_status == "UNDERPRICED_DEAL":
            deal_note = f"🔥 HIGH-VALUE DEAL: Asking price is {abs(spread_pct)}% below prevailing market value!"
        elif deal_status == "OVERPRICED":
            deal_note = f"⚠️ MARKET NOTE: Asking price is {spread_pct}% above prevailing market trend."
        else:
            deal_note = "✅ FAIR VALUE: Asking price aligns with current market trend."

        template = (
            f"Assalam-o-Alaikum {name},\n\n"
            f"Regarding your inquiry for {property_type.title()} in {society}, {city}:\n\n"
            f"• Quoted Demand: {asking_price_pkr:,} PKR\n"
        )

        if estimated_val > 0:
            template += (
                f"• AI Fair Valuation: {estimated_val:,} PKR\n"
                f"• Deal Rating: {deal_status} ({deal_note})\n\n"
            )

        if len(matches) > 0:
            template += (
                f"🤝 Active Buyer Network: We currently have {len(matches)} active buyer leads "
                f"with matching budgets in {society}.\n\n"
            )

        template += (
            "Would you be available for a brief call today to discuss proceeding?\n\n"
            "Best regards,\n"
            "DealersFlow Network"
        )

        return template

    @classmethod
    def generate_wa_link(cls, phone: Optional[str], text_message: str) -> str:
        """
        Encodes RFC-3986 compliant https://wa.me/ deep link with character truncation safety.
        """
        clean_phone = cls.sanitize_phone_number(phone)

        # Enforce URI character limit safety cap (prevents mobile browser truncation)
        if len(text_message) > cls.MAX_URI_CHAR_LIMIT:
            truncated_text = text_message[:cls.MAX_URI_CHAR_LIMIT - 3] + "..."
        else:
            truncated_text = text_message

        # RFC-3986 percent-encoding (spaces -> %20, newlines -> %0A)
        encoded_payload = urllib.parse.quote(truncated_text, safe="")
        return f"https://wa.me/{clean_phone}?text={encoded_payload}"

    
    @classmethod
    def generate_wa_web_link(cls, phone: Optional[str], text_message: str) -> str:
        """
        Direct WhatsApp Web URI: Bypasses the landing page and opens chat directly in WhatsApp Web.
        """
        clean_phone = cls.sanitize_phone_number(phone)
        if len(text_message) > cls.MAX_URI_CHAR_LIMIT:
            text_message = text_message[:cls.MAX_URI_CHAR_LIMIT - 3] + "..."
        encoded = urllib.parse.quote(text_message, safe="")
        return f"https://web.whatsapp.com/send?phone={clean_phone}&text={encoded}"
    
    @classmethod
    def dispatch_via_sandbox(cls, to_phone: str, body_text: str) -> Dict[str, Any]:
        """
        Optional fallback adapter: Dispatches automated message via Twilio WhatsApp Sandbox
        if credentials are present in environment variables.
        """
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

        if not account_sid or not auth_token:
            logger.info("[*] Twilio credentials absent. Skipping automated sandbox dispatch.")
            return {"status": "skipped", "reason": "No credentials configured"}

        try:
            from twilio.rest import Client  # type: ignore
            client = Client(account_sid, auth_token)

            clean_to = cls.sanitize_phone_number(to_phone)
            message = client.messages.create(
                from_=from_number,
                body=body_text,
                to=f"whatsapp:+{clean_to}"
            )
            return {"status": "sent", "sid": message.sid}
        except ImportError:
            logger.warning("[!] Twilio library not installed. Install via `pip install twilio`.")
            return {"status": "error", "error": "Twilio library not installed"}
        except Exception as e:
            logger.warning(f"[!] Twilio sandbox dispatch failed: {e}")
            return {"status": "error", "error": str(e)}


# Global Singleton accessor
whatsapp_service = WhatsAppService()
