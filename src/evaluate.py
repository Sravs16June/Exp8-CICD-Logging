import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from train import load_config, run_training


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Model Evaluation stage. Computes the metrics Task 4 asks for."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["notckd", "ckd"])
    return metrics, cm, report


if __name__ == "__main__":
    cfg = load_config()
    pipeline, X_train, X_test, y_train, y_test = run_training(cfg)

    metrics, cm, report = evaluate_model(pipeline, X_test, y_test)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    Path(cfg["paths"]["metrics_output"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["paths"]["metrics_output"], "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm}, f, indent=2)
    print("\nSaved metrics to:", cfg["paths"]["metrics_output"])