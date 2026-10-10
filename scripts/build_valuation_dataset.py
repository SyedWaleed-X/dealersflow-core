import random
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# GROUND-TRUTH VALUATION MATRIX: 45+ REGIONAL SOCIETIES & PHASES (PKR / SQ FT)
# Calibrated against active listings, file rates, and sector transaction data.
# ==============================================================================
PRESTIGE_RATES = {
    # --- LAHORE ---
    ("Lahore", "DHA", "Phase 6"): 13500,         # 1 Kanal ~ 6.0 Crore
    ("Lahore", "DHA", "Phase 5"): 13000,         # 1 Kanal ~ 5.8 Crore
    ("Lahore", "DHA", "Phase 7"): 8500,          # 1 Kanal ~ 3.8 Crore
    ("Lahore", "DHA", "Phase 8"): 12500,         # 1 Kanal ~ 5.6 Crore
    ("Lahore", "DHA", "Phase 9 Prism"): 6000,    # 1 Kanal ~ 2.7 Crore, 5M ~ 68 Lac
    ("Lahore", "DHA", "Phase 10"): 3000,         # File rate ~ 1.35 Crore/Kanal
    ("Lahore", "DHA", "General"): 9000,
    ("Lahore", "Bahria Town", "Sector A"): 7500, # 1 Kanal ~ 3.3 Crore
    ("Lahore", "Bahria Town", "Sector B"): 7000,
    ("Lahore", "Bahria Town", "Sector C"): 6500, # 10 Marla ~ 1.45 Crore
    ("Lahore", "Bahria Town", "Sector E"): 6000,
    ("Lahore", "Bahria Town", "Sector F"): 4500, # 5 Marla ~ 50 Lac
    ("Lahore", "Bahria Town", "General"): 6000,
    ("Lahore", "Johar Town", "Block B"): 8500,   # 5 Marla ~ 95 Lac
    ("Lahore", "Johar Town", "General"): 8000,
    ("Lahore", "Gulberg III", "General"): 18000, # Commercial/Luxury
    ("Lahore", "Lake City", "Sector M-8"): 4800, # 5 Marla ~ 54 Lac
    ("Lahore", "Etihad Town", "Phase 1"): 5500,

    # --- ISLAMABAD & RAWALPINDI ---
    ("Islamabad", "Sector F-6", "General"): 24000,# Luxury capital hub
    ("Islamabad", "Sector F-7", "Phase 2"): 22000,# 1 Kanal ~ 9.9 Crore
    ("Islamabad", "Sector F-10", "General"): 16000,
    ("Islamabad", "Sector F-11", "General"): 14000,
    ("Islamabad", "Sector G-11", "General"): 9000,# 1 Kanal ~ 4.0 Crore, 5M ~ 1.0 Crore
    ("Islamabad", "Sector G-13", "General"): 7500,
    ("Islamabad", "Sector G-14", "General"): 6500,
    ("Islamabad", "DHA Phase 2", "Sector C"): 11000,# 1 Kanal ~ 4.9 Crore
    ("Islamabad", "DHA Phase 2", "General"): 10000,
    ("Islamabad", "Bahria Enclave", "Sector M"): 6500,# 5 Marla ~ 73 Lac
    ("Islamabad", "WAPDA Town", "Block C"): 4500, # 10 Marla ~ 1.0 Crore
    ("Islamabad", "B-17", "Block A"): 5000,       # 1 Kanal ~ 2.2 Crore
    ("Islamabad", "Margalla Enclave", "General"): 7500,
    ("Rawalpindi", "Bahria Town", "Phase 8"): 5500,
    ("Rawalpindi", "Bahria Town", "Phase 4"): 6500,
    ("Rawalpindi", "Chaklala Scheme 3", "General"): 7500,
    ("Rawalpindi", "Satellite Town", "General"): 6000,

    # --- GUJRANWALA ---
    ("Gujranwala", "Citi Housing", "Phase 1"): 4800,   # 5 Marla ~ 54 Lac, 10M ~ 1.08 Cr
    ("Gujranwala", "Citi Housing", "Phase 2"): 4200,
    ("Gujranwala", "Citi Housing", "General"): 4500,
    ("Gujranwala", "DC Colony", "Block A"): 4200,      # 10 Marla ~ 95 Lac
    ("Gujranwala", "DC Colony", "General"): 3900,
    ("Gujranwala", "Royal Palm City", "Block B"): 4500,# 5 Marla ~ 50 Lac
    ("Gujranwala", "Royal Palm City", "General"): 4200,
    ("Gujranwala", "Master City", "General"): 3400,    # 5 Marla ~ 38 Lac, 10M ~ 76 Lac
    ("Gujranwala", "Wapda City", "General"): 3200,     # Possession plot rate
    ("Gujranwala", "Fazaia Housing", "General"): 3500, # 20 Marla (Kanal) ~ 1.55 Crore
    ("Gujranwala", "DHA Sector G", "General"): 5500,   # 5 Marla ~ 62 Lac

    # --- KARACHI ---
    ("Karachi", "DHA", "Phase 8"): 22000,       # 500 Sq Yds ~ 9.9 Crore
    ("Karachi", "DHA", "Phase 6"): 20000,
    ("Karachi", "Clifton", "Block 4"): 21000,
    ("Karachi", "Gulshan-e-Iqbal", "Block 13D"): 6500,# 240 Sq Yds ~ 1.4 Crore
    ("Karachi", "Gulistan-e-Jauhar", "Block 15"): 5500,
    ("Karachi", "Bahria Town", "Precinct 1"): 4500,   # 250 Sq Yds ~ 1.0 Crore
    ("Karachi", "Bahria Town", "Precinct 10B"): 3200, # 125 Sq Yds ~ 36 Lac
    ("Karachi", "Bahria Town", "Precinct 4"): 3500,   # 500 Sq Yds ~ 1.5 Crore

    # --- MULTAN & FAISALABAD ---
    ("Multan", "DHA", "A-Block"): 3000,         # 1 Kanal ~ 1.35 Crore
    ("Multan", "DHA", "General"): 2400,         # 1 Kanal ~ 1.08 Crore, 5M ~ 27 Lac
    ("Multan", "Wapda Town", "Phase 1"): 3800,
    ("Multan", "Airport Road", "General"): 5000,
    ("Faisalabad", "Citi Housing", "General"): 4000,
}

PROPERTY_TYPES = ["PLOT", "HOUSE", "COMMERCIAL"]

def generate_grand_dataset(num_records: int = 20000) -> pd.DataFrame:
    print(f"[*] Generating {num_records:,} realistic transactions across all Pakistan metros...")
    rows = []
    locations = list(PRESTIGE_RATES.keys())

    for _ in range(num_records):
        city, society, phase = random.choice(locations)
        prop_type = random.choice(PROPERTY_TYPES)

        # Standard Pakistani cuttings in sqft:
        # 3 Marla (675), 5 Marla (1125), 7 Marla (1575), 10 Marla (2250), 
        # 1 Kanal (4500), 2 Kanal (9000), 125 Sq Yd (1125), 250 Sq Yd (2250), 500 Sq Yd (4500)
        area_sqft = float(random.choice([675, 1125, 1125, 1575, 2250, 2250, 4500, 4500, 9000]))

        bedrooms = 0
        bathrooms = 0
        if prop_type == "HOUSE":
            if area_sqft <= 1125:
                bedrooms = random.choice([3, 3, 4])
            elif area_sqft <= 2250:
                bedrooms = random.choice([4, 5, 5])
            else:
                bedrooms = random.choice([5, 6, 7])
            bathrooms = bedrooms + random.choice([0, 1])

        # Base rate per sqft for location
        base_rate = PRESTIGE_RATES[(city, society, phase)]

        # Construction and Commercial Footfall Multipliers
        type_multiplier = 1.0
        if prop_type == "HOUSE":
            type_multiplier = 1.48  # Construction material & superstructure premium
        elif prop_type == "COMMERCIAL":
            type_multiplier = 2.25  # Commercial retail & plaza multiplier

        # Market variance (Gaussian noise +/- 7%)
        noise = np.random.normal(1.0, 0.07)

        raw_price = area_sqft * base_rate * type_multiplier * noise

        # Round to realistic dealer increments (nearest 50,000 PKR)
        price_pkr = int(round(raw_price / 50000.0) * 50000)

        rows.append({
            "city": city,
            "society": society,
            "phase": phase,
            "property_type": prop_type,
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "price_pkr": price_pkr
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_grand_dataset(20000)
    out_path = "data/valuation_train.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Master 20,000-row real estate dataset generated at: {out_path}")