from typing import Dict, Any, List

# Pre-computed golden fixtures for zero-risk presentations & offline evaluations
GOLDEN_DEMO_FIXTURES: List[Dict[str, Any]] = [
    {
        "id": "fixture_dha_lahore_commercial",
        "title": "1 Kanal Commercial Plot Sale in DHA Phase 6 Lahore",
        "raw_text": "Aoa bhai, DHA Phase 6 block CCA mein 1 kanal commercial plot bechna hai demand 8 crore 50 lac hai direct owner tariq 03001234567 urgent cash sale.",
        "expected_lead": {
            "intent": "SELL",
            "property_type": "COMMERCIAL",
            "city": "Lahore",
            "society": "DHA",
            "phase": "Phase 6",
            "sector_block": "Block CCA",
            "price_pkr": 85000000,
            "area_sqft": 4500.0,
            "client_name": "Tariq",
            "client_phone": "923001234567"
        },
        "expected_valuation": {
            "estimated_price_pkr": 68550000,
            "deal_status": "OVERPRICED",
            "spread_percentage": 24.0
        }
    },
    {
        "id": "fixture_fazaia_gujranwala_buyer",
        "title": "20 Marla Plot Buyer Inquiry in Fazaia Housing Gujranwala",
        "raw_text": "Assalam-o-Alaikum bhai, mujhe Gujranwala Fazaia Housing Scheme mein 1 kanal (20 marla) corner plot chahiye, mera budget taqreeban 1 crore 35 lac hai.",
        "expected_lead": {
            "intent": "BUY",
            "property_type": "PLOT",
            "city": "Gujranwala",
            "society": "Fazaia Housing",
            "phase": "General",
            "price_pkr": 13500000,
            "area_sqft": 4500.0,
            "client_name": "Chaudhry Akram",
            "client_phone": "923219876543"
        },
        "expected_valuation": {
            "estimated_price_pkr": 13500000,
            "deal_status": "FAIR_MARKET",
            "spread_percentage": 0.0
        }
    },
    {
        "id": "fixture_islamabad_luxury_house",
        "title": "2 Kanal Luxury Villa Buyer in Sector F-7 Islamabad",
        "raw_text": "Mujhe Islamabad Sector F-7 mein 2 kanal luxury villa chahiye budget 24 crore tak hai Malik Sb 03015551234.",
        "expected_lead": {
            "intent": "BUY",
            "property_type": "HOUSE",
            "city": "Islamabad",
            "society": "Sector F-7",
            "phase": "Phase 2",
            "price_pkr": 240000000,
            "area_sqft": 9000.0,
            "client_name": "Malik Sb",
            "client_phone": "923015551234"
        },
        "expected_valuation": {
            "estimated_price_pkr": 261000000,
            "deal_status": "UNDERPRICED_DEAL",
            "spread_percentage": -8.0
        }
    }
]


def get_demo_fixture_by_id(fixture_id: str) -> Dict[str, Any]:
    """Retrieves a specific golden demonstration fixture."""
    for fixture in GOLDEN_DEMO_FIXTURES:
        if fixture["id"] == fixture_id:
            return fixture
    return GOLDEN_DEMO_FIXTURES[0]