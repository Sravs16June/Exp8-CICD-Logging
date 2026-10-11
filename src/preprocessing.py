import pandas as pd
import numpy as np


def validate_and_clean(df: pd.DataFrame, target_col: str = "class") -> pd.DataFrame:
    """Data Validation / Preparation stage.
    Detects and handles missing values, duplicate records, and inconsistent dtypes,
    as required by Task 1. Returns a cleaned DataFrame."""
    report = {}

    # 1. Duplicate records
    n_dupes = df.duplicated().sum()
    report["duplicate_rows_removed"] = int(n_dupes)
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Missing values — numeric columns get median imputation (robust to outliers)
    missing_before = df.isna().sum()
    report["missing_values_by_column"] = {
        c: int(v) for c, v in missing_before.items() if v > 0
    }
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # 3. Inconsistent data types — integer-coded columns stored as float get cast back
    int_like_cols = ["al", "su", "htn", "dm", target_col]
    for c in int_like_cols:
        if c in df.columns:
            df[c] = df[c].round().astype(int)

    report["final_shape"] = df.shape
    return df, report


def split_features_target(df: pd.DataFrame, target_col: str = "class"):
    """Separates input features from the target variable."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset("data/ckd_raw.csv")
    print("Before cleaning:", df.shape, "| duplicates:", df.duplicated().sum(),
          "| missing:", df.isna().sum().sum())

    clean_df, report = validate_and_clean(df)
    print("\nCleaning report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    clean_df.to_csv("data/ckd_clean.csv", index=False)
    print("\nSaved cleaned dataset to data/ckd_clean.csv")