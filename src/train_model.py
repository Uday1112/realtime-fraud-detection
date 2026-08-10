"""
Trains an unsupervised Isolation Forest anomaly detector on the historical
transaction features and evaluates it against the injected fraud labels.

The model never sees `is_fraud_synthetic` during fitting — labels are used
only afterward, to measure how well an unsupervised detector recovers known
injected patterns. This mirrors real fraud detection: confirmed-fraud labels
are scarce/delayed in production, so detection has to work without them, but
you still need a way to validate the detector actually catches anything.

Run with: python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_parquet(DATA_DIR / "historical_transactions.parquet")
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud_synthetic"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    contamination = float(y_train.mean())  # anomaly rate observed in training data
    print(f"training IsolationForest (contamination={contamination:.4f}) on {len(X_train)} rows...")

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)  # unsupervised: labels are not passed in

    # IsolationForest: predict() gives -1 (anomaly) / 1 (normal);
    # decision_function() gives a continuous score, lower = more anomalous.
    raw_pred = model.predict(X_test_scaled)
    y_pred = (raw_pred == -1).astype(int)
    anomaly_score = -model.decision_function(X_test_scaled)  # flip sign: higher = more anomalous

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, anomaly_score)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=["normal", "fraud_pattern"], zero_division=0)

    # Recall broken down by injected fraud pattern — an aggregate number
    # hides that some patterns (e.g. velocity bursts) are far easier for an
    # unsupervised detector to isolate than others (e.g. a modest amount
    # spike for an already-high-variance spender).
    pattern_series = df.loc[X_test.index, "fraud_pattern"]
    per_pattern = []
    for pattern in ["amount_spike", "velocity_burst", "odd_hour_bulk"]:
        mask = pattern_series == pattern
        n = int(mask.sum())
        caught = int((y_pred[mask.values] == 1).sum()) if n else 0
        per_pattern.append((pattern, n, caught, caught / n if n else 0.0))

    print(report_text)
    print(f"ROC-AUC: {auc:.4f}")
    print("Recall by fraud pattern:")
    for pattern, n, caught, rate in per_pattern:
        print(f"  {pattern:16s} {caught}/{n} caught ({rate:.1%})")

    joblib.dump(model, MODELS_DIR / "anomaly_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "contamination_used": contamination,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "feature_columns": FEATURE_COLUMNS,
        "recall_by_pattern": {p: {"n": n, "caught": c, "rate": r} for p, n, c, r in per_pattern},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        "# Model Evaluation — Isolation Forest Fraud/Anomaly Detector",
        "",
        f"Trained unsupervised (no labels used in fitting) on {len(X_train):,} historical "
        f"transactions, evaluated against injected synthetic fraud-pattern labels on a held-out "
        f"{len(X_test):,}-row test split.",
        "",
        "## Results",
        f"- **Precision**: {precision:.1%} — of transactions flagged as anomalous, this share were actual injected fraud patterns",
        f"- **Recall**: {recall:.1%} — of actual injected fraud patterns, this share were caught",
        f"- **F1**: {f1:.3f}",
        f"- **ROC-AUC**: {auc:.3f}",
        "",
        "## Recall by injected fraud pattern",
        "| Pattern | Caught / Total | Recall |",
        "|---|---|---|",
    ] + [f"| {p} | {c}/{n} | {r:.1%} |" for p, n, c, r in per_pattern] + [
        "",
        "## Confusion matrix (test set)",
        "| | Predicted normal | Predicted fraud_pattern |",
        "|---|---|---|",
        f"| Actual normal | {cm[0,0]} | {cm[0,1]} |",
        f"| Actual fraud_pattern | {cm[1,0]} | {cm[1,1]} |",
        "",
        "## Features used",
        "".join(f"- `{c}`\n" for c in FEATURE_COLUMNS),
        "## Classification report",
        "```",
        report_text,
        "```",
    ]
    (REPORTS_DIR / "model_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORTS_DIR / 'model_evaluation.md'} and {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
