import random
import numpy as np
import pandas as pd
from pathlib import Path

# Base price per square foot across cities and societies (PKR)
PRESTIGE_RATES = {
    # Lahore
    ("Lahore", "DHA", "Phase 6"): 20000,
    ("Lahore", "DHA", "Phase 5"): 19000,
    ("Lahore", "Bahria Town", "Sector C"): 11000,
    ("Lahore", "Johar Town", "Block B"): 14000,
    ("Lahore", "Gulberg III", "General"): 24000,
    # Islamabad
    ("Islamabad", "Sector F-7", "Phase 2"): 28000,
    ("Islamabad", "DHA Phase 2", "Sector C"): 17000,
    ("Islamabad", "Bahria Enclave", "Sector M"): 12000,
    ("Islamabad", "WAPDA Town", "Block C"): 9000,
    # Gujranwala
    ("Gujranwala", "Citi Housing", "Phase 1"): 9500,
    ("Gujranwala", "DC Colony", "Block A"): 8500,
    ("Gujranwala", "Royal Palm City", "Block B"): 7500,
    ("Gujranwala", "Master City", "General"): 6500,
    # Karachi
    ("Karachi", "DHA", "Phase 8"): 27000,
    ("Karachi", "Clifton", "Block 4"): 26000,
    ("Karachi", "Gulshan-e-Iqbal", "Block 13D"): 10000,
    ("Karachi", "Bahria Town", "Precinct 1"): 9000,
    # Rawalpindi
    ("Rawalpindi", "Bahria Town", "Phase 8"): 11000,
    ("Rawalpindi", "Chaklala Scheme 3", "General"): 13000,
}

PROPERTY_TYPES = ["PLOT", "HOUSE", "COMMERCIAL"]

def generate_dataset(num_records: int = 5000) -> pd.DataFrame:
    print(f"[*] Generating {num_records} realistic Pakistani real estate transactions...")
    rows = []

    locations = list(PRESTIGE_RATES.keys())

    for _ in range(num_records):
        city, society, phase = random.choice(locations)
        prop_type = random.choice(PROPERTY_TYPES)

        # Realistic area in sqft: 5 Marla (1125), 10 Marla (2250), 1 Kanal (4500), 2 Kanal (9000)
        area_sqft = float(random.choice([1125, 1125, 2250, 2250, 4500, 9000]))

        bedrooms = 0
        bathrooms = 0
        if prop_type == "HOUSE":
            bedrooms = 3 if area_sqft <= 1125 else (4 if area_sqft <= 2250 else random.randint(5, 7))
            bathrooms = bedrooms + random.choice([0, 1])

        # Base valuation formula based on location rate per sqft
        base_rate = PRESTIGE_RATES[(city, society, phase)]
        
        # Property type multipliers
        type_multiplier = 1.0
        if prop_type == "HOUSE":
            type_multiplier = 1.45  # Construction cost premium
        elif prop_type == "COMMERCIAL":
            type_multiplier = 2.10  # Commercial footfall multiplier

        # Market variance (Gaussian noise +/- 8%)
        noise = np.random.normal(1.0, 0.08)

        # Calculate price in PKR
        raw_price = area_sqft * base_rate * type_multiplier * noise
        # Round to nearest 50,000 PKR
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
    dataset_df = generate_dataset(5000)
    output_path = "data/valuation_train.csv"
    dataset_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Valuation training dataset saved to: {output_path} ({len(dataset_df)} rows)")