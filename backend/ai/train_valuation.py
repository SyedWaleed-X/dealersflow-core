import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

DATA_PATH = Path("data/valuation_train.csv")
MODEL_DIR = Path("models")
MODEL_OUTPUT_PATH = MODEL_DIR / "catboost_valuation.cbm"

# Categorical columns passed directly to CatBoost without one-hot encoding
CAT_FEATURES = ["city", "society", "phase", "property_type"]
NUM_FEATURES = ["area_sqft", "bedrooms", "bathrooms"]
FEATURE_COLUMNS = CAT_FEATURES + NUM_FEATURES

def train():
    print("=" * 75)
    print("CATBOOST VALUATION REGRESSION MODEL TRAINING")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}. Run scripts/build_valuation_dataset.py first.")

    print(f"[*] Loading training corpus from: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Ensure string typing on categorical features to prevent type mismatch
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str)

    X = df[FEATURE_COLUMNS]
    # Log-transform target to compress multi-Crore variance
    y = np.log1p(df["price_pkr"])

    # 2. 85% Train / 15% Holdout Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"[*] Training Set: {len(X_train)} samples | Validation Set: {len(X_val)} samples")
    print(f"[*] Native Categorical Features: {CAT_FEATURES}")

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)

    # 3. Initialize CatBoost Regressor
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.06,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100  # Log every 100 iterations
    )

    print("\n[*] Training CatBoost Trees with Ordered Target Statistics...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # 4. Save Model in CBM binary format
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODEL_OUTPUT_PATH), format="cbm")
    print(f"\n[SUCCESS] Model successfully saved to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train()