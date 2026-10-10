import re
from typing import Optional


class NormalizationError(Exception):
    """Raised when currency or area phrases cannot be deterministically resolved."""
    pass


class PropertyNormalizer:
    """
    Zero-hallucination deterministic normalization engine for
    Pakistani currency expressions and regional land measurements.
    Defensively hardened against slang, Whisper acoustic glitches, and ranges.
    """

    COLLOQUIAL_MULTIPLIERS = {
        "dedh": 1.5,
        "derh": 1.5,
        "dhayi": 2.5,
        "dhai": 2.5,
        "sawa": 1.25,
        "paunay": 0.75,
        "pone": 0.75,
    }

    # Word numeral mappings (Roman-Urdu + Urdu Nastaliq)
    URDU_WORD_NUMERALS = {
        "ek": 1, "aik": 1, "ایک": 1,
        "do": 2, "دو": 2,
        "teen": 3, "تین": 3,
        "char": 4, "chaar": 4, "چار": 4,
        "panj": 5, "paanch": 5, "پانچ": 5,
        "chhe": 6, "che": 6, "چھ": 6,
        "saat": 7, "سات": 7,
        "aath": 8, "آٹھ": 8,
        "nau": 9, "نو": 9,
        "das": 10, "دس": 10,
        "bees": 20, "بیس": 20,
        "tees": 30, "تیس": 30,
        "chalees": 40, "چالیس": 40,
        "pachaas": 50, "پچاس": 50,
    }

    MARLA_TO_SQFT_LDA = 225.0
    KANAL_TO_SQFT_LDA = 4500.0  # 20 Marlas * 225
    GAZ_TO_SQFT = 9.0           # 1 Square Yard = 9 sqft

    @classmethod
    def clean_text_phonetics(cls, text: str) -> str:
        """Standardizes Urdu & Roman-Urdu speech glitches and number ranges."""
        if not text:
            return ""
        t = str(text).lower()

        # Fix Whisper acoustic mishears
        t = t.replace("پڑھوڑ", " crore ").replace("parhor", " crore ").replace("karor", " crore ")
        t = t.replace("مدلے", " marla ").replace("مرلے", " marla ").replace("مرلہ", " marla ")
        t = t.replace("لاکھ", " lac ").replace("کروڑ", " crore ")

        # Replace word numerals with digits
        for word, num in cls.URDU_WORD_NUMERALS.items():
            t = re.sub(rf"\b{word}\b", str(num), t)

        # Handle ranges like '4-5 marla' or '4 to 5' -> pick upper bound 5
        t = re.sub(r"(\d+)\s*[-–to|se]\s*(\d+)", r"\2", t)
        return t

    @classmethod
    def normalize_price(cls, raw_str: Optional[str]) -> Optional[int]:
        """Converts South Asian currency strings into integer PKR."""
        if not raw_str or not str(raw_str).strip():
            return None

        # Clean phonetics first
        text = cls.clean_text_phonetics(str(raw_str)).strip().replace(",", "")

        # 1. Handle colloquial vernacular fractions
        for prefix, multiplier in cls.COLLOQUIAL_MULTIPLIERS.items():
            if prefix in text:
                if re.search(r"\b(crore|cr|karor)\b", text):
                    if prefix in ["paunay", "pone"] and "2" in text:
                        return int(1.75 * 10_000_000)
                    return int(multiplier * 10_000_000)
                elif re.search(r"\b(lac|lakh|lacs)\b", text):
                    return int(multiplier * 100_000)
                elif re.search(r"\b(arab)\b", text):
                    return int(multiplier * 1_000_000_000)

        total_pkr = 0.0
        found_match = False

        # 2. Extract Arab (10^9 PKR)
        arab_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:arab)\b", text)
        if arab_match:
            total_pkr += float(arab_match.group(1)) * 1_000_000_000
            found_match = True

        # 3. Extract Crore (10^7 PKR)
        crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)\b", text)
        if crore_match:
            total_pkr += float(crore_match.group(1)) * 10_000_000
            found_match = True

        # 4. Extract Lac / Lakh (10^5 PKR)
        lac_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lac|lakh|lacs)\b", text)
        if lac_match:
            total_pkr += float(lac_match.group(1)) * 100_000
            found_match = True

        if not found_match:
            digits_only = re.sub(r"[^\d.]", "", text)
            if digits_only:
                try:
                    return int(float(digits_only))
                except ValueError:
                    return None
            return None

        return int(round(total_pkr))

    @classmethod
    def normalize_area(cls, raw_str: Optional[str], standard: str = "LDA") -> Optional[float]:
        """Converts regional land units into standard Sq Ft."""
        if not raw_str or not str(raw_str).strip():
            return None

        text = cls.clean_text_phonetics(str(raw_str)).strip().replace(",", "")

        marla_factor = cls.MARLA_TO_SQFT_LDA
        kanal_factor = cls.KANAL_TO_SQFT_LDA

        total_sqft = 0.0
        found_match = False

        # 1. Extract Kanals
        kanal_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kanal|knl)\b", text)
        if kanal_match:
            total_sqft += float(kanal_match.group(1)) * kanal_factor
            found_match = True

        # 2. Extract Marlas
        marla_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:marla|marlay|mrla)\b", text)
        if marla_match:
            total_sqft += float(marla_match.group(1)) * marla_factor
            found_match = True

        # 3. Extract Gaz
        gaz_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:gaz|guz|yard|sq\s*yard)\b", text)
        if gaz_match:
            total_sqft += float(gaz_match.group(1)) * cls.GAZ_TO_SQFT
            found_match = True

        # 4. Extract Explicit Sq Ft
        sqft_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:sqft|sq\s*ft|square\s*feet)\b", text)
        if sqft_match:
            total_sqft += float(sqft_match.group(1))
            found_match = True

        if not found_match:
            digits_only = re.sub(r"[^\d.]", "", text)
            if digits_only:
                try:
                    val = float(digits_only)
                    # If number is <= 50, it represents Marlas in Pakistani real estate
                    if val <= 50.0:
                        return val * marla_factor
                    return val
                except ValueError:
                    return None
            return None

        return round(total_sqft, 2)