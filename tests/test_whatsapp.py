import pytest
from backend.services.whatsapp import WhatsAppService, whatsapp_service


# ==========================================
# TEST 1: PAKISTANI & INTERNATIONAL PHONE NORMALIZATION
# ==========================================
@pytest.mark.parametrize("raw_input, expected_msisdn", [
    # Standard Pakistani local format
    ("03001234567", "923001234567"),
    ("0321-9876543", "923219876543"),
    ("0333 5556677", "923335556677"),
    ("0345.1112233", "923451112233"),
    # With international +92 code
    ("+923001234567", "923001234567"),
    ("+92 321 9876543", "923219876543"),
    ("923001234567", "923001234567"),
    # Messy spacing and brackets
    ("(0300) 123-4567", "923001234567"),
    ("0300 12 34 567", "923001234567"),
    # Fallback cases
    ("", "923001234567"),
    (None, "923001234567"),
])
def test_phone_number_sanitization(raw_input, expected_msisdn):
    result = WhatsAppService.sanitize_phone_number(raw_input)
    assert result == expected_msisdn, f"Failed for '{raw_input}': got {result}, expected {expected_msisdn}"
    # Verify no illegal characters exist for wa.me protocol
    assert "+" not in result
    assert "-" not in result
    assert " " not in result


# ==========================================
# TEST 2: TEMPLATE GENERATION & NEGOTIATION CONTEXT
# ==========================================
def test_negotiation_template_generation():
    template = WhatsAppService.generate_negotiation_template(
        client_name="Tariq Malik",
        property_type="COMMERCIAL",
        society="DHA Phase 6",
        city="Lahore",
        asking_price_pkr=85_000_000,
        valuation_dict={
            "estimated_price_pkr": 68_550_000,
            "deal_status": "OVERPRICED",
            "spread_percentage": 24.0
        },
        matched_leads=[{"id": "1"}, {"id": "2"}, {"id": "3"}]
    )

    assert "Tariq Malik" in template
    assert "85,000,000 PKR" in template
    assert "68,550,000 PKR" in template
    assert "OVERPRICED" in template
    assert "3 active buyer leads" in template
    print(f"\n[+] Negotiation Template Generated:\n{template}")


# ==========================================
# CHAOS TEST 12.1: URI CHARACTER LIMIT TRUNCATION
# ==========================================
def test_uri_character_limit_truncation():
    """
    Simulates a massive 3,000-character description passed to the deep link engine.
    Asserts that the message payload is safely capped at 1,500 characters to prevent
    mobile browser URL dropping.
    """
    massive_text = "DHA Phase 6 Commercial Plot Description: " + ("Very good prime location. " * 150)
    assert len(massive_text) > 2500

    link = WhatsAppService.generate_wa_link("03001234567", massive_text)
    assert "https://wa.me/923001234567?text=" in link

    # Extract decoded query text length
    import urllib.parse
    parsed_url = urllib.parse.urlparse(link)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    decoded_message = query_params["text"][0]

    assert len(decoded_message) <= 1500, f"Payload exceeded 1500 char cap: {len(decoded_message)}"
    assert decoded_message.endswith("..."), "Truncated payload missing ellipsis indicator!"
    print(f"\n[SUCCESS] Chaos Test 12.1: 3,000-char payload safely capped to {len(decoded_message)} characters.")


# ==========================================
# CHAOS TEST 12.2: CROSS-SITE SCRIPTING (XSS) SANITIZATION
# ==========================================
def test_xss_injection_sanitization():
    """
    Injects malicious HTML/JavaScript tags into client name and text.
    Verifies that RFC-3986 percent-encoding neuters all executable scripts.
    """
    xss_payload = "<script>alert('HACKED')</script> & <img src=x onerror=alert(1)>"
    link = WhatsAppService.generate_wa_link("03001234567", xss_payload)

    # Angle brackets must be percent-encoded (%3C and %3E), NEVER raw
    assert "<script>" not in link
    assert "</script>" not in link
    assert "%3Cscript%3E" in link or "%3Cscript" in link
    print("\n[SUCCESS] Chaos Test 12.2: XSS script tags neutralized via RFC-3986 percent-encoding.")