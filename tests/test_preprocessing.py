import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import generate_ckd_dataset, load_dataset
from preprocessing import validate_and_clean, split_features_target


def test_validate_and_clean_removes_duplicates_and_missing(tmp_path):
    out_path = tmp_path / "test_raw.csv"
    generate_ckd_dataset(str(out_path), n=100, seed=1)
    df = load_dataset(str(out_path))

    clean_df, report = validate_and_clean(df)

    assert clean_df.duplicated().sum() == 0
    assert clean_df.isna().sum().sum() == 0
    assert report["duplicate_rows_removed"] > 0


def test_split_features_target_shapes():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "class": [0, 1]})
    X, y = split_features_target(df)
    assert "class" not in X.columns
    assert list(y) == [0, 1]