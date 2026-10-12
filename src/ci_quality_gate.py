import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import generate_ckd_dataset, load_dataset
from preprocessing import validate_and_clean, split_features_target
from feature_engineering import engineer_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---- Quality gate threshold (Task 4) ----
MIN_ACCEPTABLE_ACCURACY = 0.70
MIN_ACCEPTABLE_ROC_AUC = 0.75


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_quality_gate():
    cfg = load_config()

    df = generate_ckd_dataset(cfg["data"]["raw_path"], seed=cfg["split"]["random_state"])
    df = load_dataset(cfg["data"]["raw_path"])
    clean_df, _ = validate_and_clean(df, cfg["data"]["target_column"])

    X, y = split_features_target(clean_df, cfg["data"]["target_column"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )

    pipeline = Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=cfg["model"]["random_state"],
        )),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    passed_accuracy = metrics["accuracy"] >= MIN_ACCEPTABLE_ACCURACY
    passed_roc_auc = metrics["roc_auc"] >= MIN_ACCEPTABLE_ROC_AUC
    gate_passed = passed_accuracy and passed_roc_auc

    result = {
        "metrics": metrics,
        "thresholds": {
            "min_accuracy": MIN_ACCEPTABLE_ACCURACY,
            "min_roc_auc": MIN_ACCEPTABLE_ROC_AUC,
        },
        "checks": {
            "accuracy_check": "PASS" if passed_accuracy else "FAIL",
            "roc_auc_check": "PASS" if passed_roc_auc else "FAIL",
        },
        "quality_gate_decision": "PASS" if gate_passed else "FAIL",
    }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/quality_gate_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 55)
    print("MODEL QUALITY GATE")
    print("=" * 55)
    print(json.dumps(result, indent=2))
    print("=" * 55)

    if gate_passed:
        print("QUALITY GATE: PASSED \u2014 model meets release criteria.")
        joblib_save(pipeline, cfg)
        sys.exit(0)
    else:
        print("QUALITY GATE: FAILED \u2014 blocking pipeline from proceeding to build/deploy.")
        sys.exit(1)


def joblib_save(pipeline, cfg):
    import joblib
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/ckd_pipeline.pkl")
    print("Validated model saved to models/ckd_pipeline.pkl")


if __name__ == "__main__":
    run_quality_gate()