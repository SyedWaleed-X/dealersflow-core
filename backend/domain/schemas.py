from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator


class PropertyIntent(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PropertyClassification(str, Enum):
    PLOT = "PLOT"
    HOUSE = "HOUSE"
    COMMERCIAL = "COMMERCIAL"
    AGRICULTURAL = "AGRICULTURAL"


class ExtractedPropertyLead(BaseModel):
    """
    Completely resilient extraction schema:
    Accepts null values for ANY field without throwing Pydantic validation errors.
    """
    intent: PropertyIntent = Field(default=PropertyIntent.BUY)
    property_type: PropertyClassification = Field(default=PropertyClassification.PLOT)
    city: Optional[str] = Field(default="Islamabad")
    society: Optional[str] = Field(default="General Area")
    phase: Optional[str] = None
    sector_block: Optional[str] = None
    raw_price: Optional[str] = None
    raw_area: Optional[str] = Field(default="5 Marla")
    bedrooms: Optional[int] = 0
    bathrooms: Optional[int] = 0
    qualitative_features: List[str] = Field(default_factory=list)
    client_name: Optional[str] = None
    client_phone: Optional[str] = None

    @field_validator("city", "society", "raw_area", mode="before")
    @classmethod
    def handle_none_strings(cls, v: Any, info) -> str:
        """Converts null inputs from LLM into safe fallback strings."""
        if not v or str(v).strip().lower() in ["none", "null", ""]:
            if info.field_name == "city":
                return "Islamabad"
            if info.field_name == "society":
                return "General Area"
            if info.field_name == "raw_area":
                return "5 Marla"
        return str(v).strip()


class NormalizedPropertyLead(BaseModel):
    """Normalized schema with verified mathematical values."""
    intent: PropertyIntent
    property_type: PropertyClassification
    city: str
    society: str
    phase: Optional[str] = None
    sector_block: Optional[str] = None
    raw_price: Optional[str] = None
    price_pkr: int = Field(default=15_000_000)
    raw_area: str = "5 Marla"
    area_sqft: float = Field(default=1125.0)
    bedrooms: Optional[int] = 0
    bathrooms: Optional[int] = 0
    qualitative_features: List[str] = Field(default_factory=list)
    client_name: Optional[str] = None
    client_phone: Optional[str] = None

    @classmethod
    def from_extracted_lead(cls, extracted: ExtractedPropertyLead) -> "NormalizedPropertyLead":
        from backend.domain.normalizer import PropertyNormalizer

        norm_price = PropertyNormalizer.normalize_price(extracted.raw_price) if extracted.raw_price else None
        norm_area = PropertyNormalizer.normalize_area(extracted.raw_area) if extracted.raw_area else 1125.0

        # If user didn't mention price, estimate baseline from city & area
        if not norm_price:
            estimated_marlas = (norm_area or 1125.0) / 225.0
            norm_price = int(max(3_000_000, estimated_marlas * 1_600_000))

        return cls(
            intent=extracted.intent or PropertyIntent.BUY,
            property_type=extracted.property_type or PropertyClassification.PLOT,
            city=extracted.city or "Islamabad",
            society=extracted.society or "General Area",
            phase=extracted.phase,
            sector_block=extracted.sector_block,
            raw_price=extracted.raw_price or f"{norm_price:,} PKR (Market Est.)",
            price_pkr=norm_price,
            raw_area=extracted.raw_area or "5 Marla",
            area_sqft=norm_area or 1125.0,
            bedrooms=extracted.bedrooms or 0,
            bathrooms=extracted.bathrooms or 0,
            qualitative_features=extracted.qualitative_features or [],
            client_name=extracted.client_name,
            client_phone=extracted.client_phone,
        )