import pandas as pd
import numpy as np


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering stage.
    Adds clinically meaningful derived features on top of the raw CKD measurements.
    Must be applied identically to training and inference data."""
    X = X.copy()

    # 1. BUN/Creatinine ratio — a standard clinical indicator of kidney function
    X["bu_sc_ratio"] = X["bu"] / X["sc"].replace(0, np.nan)
    X["bu_sc_ratio"] = X["bu_sc_ratio"].fillna(X["bu_sc_ratio"].median())

    # 2. Glucose-to-sodium ratio — a simple derived metabolic indicator
    X["bgr_sod_ratio"] = X["bgr"] / X["sod"].replace(0, np.nan)
    X["bgr_sod_ratio"] = X["bgr_sod_ratio"].fillna(X["bgr_sod_ratio"].median())

    # 3. Risk-flag count — how many individual clinical markers are in an abnormal range
    flags = pd.DataFrame({
        "flag_low_hemoglobin": (X["hemo"] < 11).astype(int),
        "flag_high_creatinine": (X["sc"] > 1.5).astype(int),
        "flag_high_albumin": (X["al"] >= 2).astype(int),
        "flag_high_urea": (X["bu"] > 60).astype(int),
        "flag_low_sg": (X["sg"] <= 1.010).astype(int),
    })
    X["risk_flag_count"] = flags.sum(axis=1)

    # 4. Comorbidity count — hypertension + diabetes present
    X["comorbidity_count"] = X["htn"] + X["dm"]

    return X


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import validate_and_clean, split_features_target

    df = load_dataset("data/ckd_clean.csv")
    X, y = split_features_target(df)

    X_fe = engineer_features(X)
    print("Original feature count:", X.shape[1])
    print("Feature count after engineering:", X_fe.shape[1])
    print("\nNew columns added:", [c for c in X_fe.columns if c not in X.columns])
    print("\nSample engineered rows:")
    print(X_fe[["bu_sc_ratio", "bgr_sod_ratio", "risk_flag_count", "comorbidity_count"]].head())