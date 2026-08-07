"""Training, evaluation and persistence for churn models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET


def build_pipeline(classifier: Any) -> Pipeline:
    """Build a leakage-safe preprocessing and classification pipeline."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _metrics(y_true: pd.Series, probability: Any, threshold: float = 0.5) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 4),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 4),
    }


def train_and_evaluate(
    data: pd.DataFrame,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Compare candidate models and return the best pipeline by ROC-AUC."""
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET],
        test_size=0.25,
        stratify=data[TARGET],
        random_state=random_state,
    )
    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        results[name] = _metrics(y_test, probability)
        fitted[name] = pipeline

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    report: dict[str, Any] = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "test_rows": len(x_test),
        "churn_rate": round(float(data[TARGET].mean()), 4),
        "models": results,
    }

    if model_path:
        output = Path(model_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": fitted[best_name], "report": report}, output)
    if metrics_path:
        output = Path(metrics_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return fitted[best_name], report


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    """Load a persisted model artifact."""
    return joblib.load(model_path)


def predict_customer(pipeline: Pipeline, customer: dict[str, Any]) -> dict[str, Any]:
    """Predict churn risk for one customer and attach an action label."""
    row = pd.DataFrame([customer], columns=FEATURE_COLUMNS)
    probability = float(pipeline.predict_proba(row)[0, 1])
    if probability >= 0.70:
        risk, action = "High", "Contact the customer and offer a retention incentive."
    elif probability >= 0.40:
        risk, action = "Medium", "Review recent support issues and monitor the account."
    else:
        risk, action = "Low", "No immediate intervention is required."
    return {"churn_probability": round(probability, 4), "risk_level": risk, "action": action}
