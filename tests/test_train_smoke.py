import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import generate_ckd_dataset
from preprocessing import validate_and_clean
from feature_engineering import engineer_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split


def test_pipeline_smoke_runs_end_to_end(tmp_path):
    """Smoke test: a tiny dataset through the full pipeline shape, fast enough for CI."""
    out_path = tmp_path / "smoke_raw.csv"
    df = generate_ckd_dataset(str(out_path), n=60, seed=1)
    clean_df, _ = validate_and_clean(df)

    X = clean_df.drop(columns=["class"])
    y = clean_df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    pipeline = Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=1)),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})