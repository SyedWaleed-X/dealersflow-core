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
    Null-safe extraction schema:
    Extracts raw strings without forcing deceptive city defaults.
    """
    intent: PropertyIntent = Field(default=PropertyIntent.BUY)
    property_type: PropertyClassification = Field(default=PropertyClassification.PLOT)
    city: Optional[str] = Field(default=None)
    society: Optional[str] = Field(default=None)
    phase: Optional[str] = None
    sector_block: Optional[str] = None
    raw_price: Optional[str] = None
    raw_area: Optional[str] = None
    bedrooms: Optional[int] = 0
    bathrooms: Optional[int] = 0
    qualitative_features: List[str] = Field(default_factory=list)
    client_name: Optional[str] = None
    client_phone: Optional[str] = None


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
        norm_area = PropertyNormalizer.normalize_area(extracted.raw_area) if extracted.raw_area else None

        # Clean city extraction: default to Lahore if completely absent
        detected_city = extracted.city or "Lahore"
        detected_society = extracted.society or "General Area"

        # If user didn't mention area, default to standard 5 Marla (1125 sqft)
        if not norm_area:
            norm_area = 1125.0
            raw_area_str = "5 Marla"
        else:
            raw_area_str = extracted.raw_area or f"{(norm_area/225):.1f} Marla"

        # If user didn't mention price, estimate median baseline: 15 Lac per Marla
        if not norm_price:
            estimated_marlas = norm_area / 225.0
            norm_price = int(max(3_000_000, estimated_marlas * 1_500_000))
            raw_price_str = f"{norm_price:,} PKR (Market Est.)"
        else:
            raw_price_str = extracted.raw_price or f"{norm_price:,} PKR"

        return cls(
            intent=extracted.intent or PropertyIntent.BUY,
            property_type=extracted.property_type or PropertyClassification.PLOT,
            city=detected_city,
            society=detected_society,
            phase=extracted.phase,
            sector_block=extracted.sector_block,
            raw_price=raw_price_str,
            price_pkr=norm_price,
            raw_area=raw_area_str,
            area_sqft=norm_area,
            bedrooms=extracted.bedrooms or 0,
            bathrooms=extracted.bathrooms or 0,
            qualitative_features=extracted.qualitative_features or [],
            client_name=extracted.client_name,
            client_phone=extracted.client_phone,
        )