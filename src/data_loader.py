import numpy as np
import pandas as pd
from pathlib import Path

def generate_ckd_dataset(path: str, n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generates a realistic CKD-style dataset and saves it as CSV.
    This stands in for the 'raw dataset' stage of the pipeline."""
    rng = np.random.default_rng(seed)

    age = rng.integers(2, 90, n)
    bp = rng.normal(76, 13, n).clip(50, 180)
    sg = rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], n)
    al = rng.integers(0, 5, n)
    su = rng.integers(0, 5, n)
    bgr = rng.normal(148, 75, n).clip(70, 490)
    bu = rng.normal(57, 50, n).clip(1.5, 391)
    sc = rng.normal(3.0, 5.7, n).clip(0.4, 76)
    sod = rng.normal(137, 10, n).clip(4.5, 163)
    pot = rng.normal(4.6, 3.2, n).clip(2.5, 47)
    hemo = rng.normal(12.5, 2.9, n).clip(3.1, 17.8)
    wbcc = rng.normal(8400, 2900, n).clip(2200, 26400)
    rbcc = rng.normal(4.7, 1.0, n).clip(2.1, 8.0)
    htn = rng.choice([0, 1], n, p=[0.6, 0.4])
    dm = rng.choice([0, 1], n, p=[0.67, 0.33])

    risk = (
        (al >= 2).astype(int) * 2 + (sc > 1.5).astype(int) * 3 + (bu > 60).astype(int) * 2 +
        (hemo < 11).astype(int) * 2 + (sg <= 1.010).astype(int) * 2 +
        htn * 1.5 + dm * 1.5 + (bgr > 200).astype(int) * 1 + rng.normal(0, 1.5, n)
    )
    label = (risk > np.percentile(risk, 50)).astype(int)

    df = pd.DataFrame({
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su, "bgr": bgr, "bu": bu,
        "sc": sc, "sod": sod, "pot": pot, "hemo": hemo, "wbcc": wbcc, "rbcc": rbcc,
        "htn": htn, "dm": dm, "class": label,
    })

    # Inject a few realistic data-quality issues, then clean them here at load time
    rng2 = np.random.default_rng(seed + 1)
    dup_idx = rng2.choice(df.index, size=5, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)          # duplicate records
    miss_idx = rng2.choice(df.index, size=10, replace=False)
    df.loc[miss_idx, "hemo"] = np.nan                                  # missing values
    df["al"] = df["al"].astype(float)                                  # inconsistent dtype on purpose

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path: str) -> pd.DataFrame:
    """Loads the dataset from disk. This is the Data Loading stage."""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = generate_ckd_dataset("data/ckd_raw.csv")
    print("Generated dataset shape:", df.shape)
    print(df.head())