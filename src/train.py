import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from data_loader import load_dataset
from preprocessing import validate_and_clean, split_features_target
from feature_engineering import engineer_features


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict, feature_names: list) -> Pipeline:
    """Builds a single sklearn Pipeline: feature engineering -> scaling -> model.
    Wrapping feature engineering in FunctionTransformer means it is fitted/applied
    as part of the same object as the model, so train and inference always match."""
    fe_step = FunctionTransformer(engineer_features, validate=False)

    preprocessor = ColumnTransformer(
        transformers=[("scale_all", StandardScaler(), slice(0, None))],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        random_state=cfg["model"]["random_state"],
    )

    pipeline = Pipeline([
        ("feature_engineering", fe_step),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return pipeline


def run_training(cfg: dict):
    df = load_dataset(cfg["data"]["clean_path"])
    X, y = split_features_target(df, cfg["data"]["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )

    pipeline = build_pipeline(cfg, X.columns.tolist())
    pipeline.fit(X_train, y_train)

    Path(cfg["paths"]["model_output"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg["paths"]["model_output"])
    joblib.dump(list(X.columns), "models/feature_order.pkl")

    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    cfg = load_config()
    pipeline, X_train, X_test, y_train, y_test = run_training(cfg)
    print("Training complete.")
    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
    print("Pipeline steps:", [name for name, _ in pipeline.steps])
    print("Saved model to:", cfg["paths"]["model_output"])