"""Training, evaluation and persistence for churn models."""

from __future__ import annotations

import json
from math import isfinite
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

from .calibration import fit_sigmoid_calibrator, probability_quality
from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from .explain import global_feature_importance
from .threshold import threshold_counts

ARTIFACT_VERSION = 5


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


def split_training_data(
    data: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve 25% for final testing; split the remainder into 60% train and 15% validation."""
    development, test = train_test_split(
        data, test_size=0.25, stratify=data[TARGET], random_state=random_state
    )
    train, validation = train_test_split(
        development, test_size=0.20,
        stratify=development[TARGET], random_state=random_state,
    )
    return train, validation, test


def train_and_evaluate(
    data: pd.DataFrame,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Select on validation ROC-AUC and evaluate on an independent final test split."""
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not data[TARGET].isin([0, 1]).all() or data[TARGET].nunique() != 2:
        raise ValueError("Churn labels must contain both 0 and 1, with no missing values.")
    train, validation, test = split_training_data(data, random_state=random_state)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation[TARGET]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET]
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

    validation_results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, Any] = {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        validation_results[name] = _metrics(y_validation, probability)
        fitted[name] = pipeline
        validation_probabilities[name] = probability

    best_name = max(
        fitted,
        key=lambda name: roc_auc_score(y_validation, validation_probabilities[name]),
    )
    # Compare a training-only calibrated candidate without replacing the active model.
    calibration_candidate = fit_sigmoid_calibrator(
        fitted[best_name], x_train, y_train, random_state=random_state,
    )
    calibrated_validation = calibration_candidate.predict_proba(x_validation)[:, 1]
    # The test split is first scored after the selected model is fixed.
    test_probabilities = {
        name: pipeline.predict_proba(x_test)[:, 1] for name, pipeline in fitted.items()
    }
    results = {name: _metrics(y_test, scores) for name, scores in test_probabilities.items()}
    calibration_report = {
        "status": "diagnostic_only",
        "base_model": best_name,
        "active_variant": "original",
        "method": "sigmoid",
        "folds": calibration_candidate.cv.n_splits,
        "fit_source": "training_out_of_fold",
        "training_rows": len(train),
        "bin_strategy": "uniform",
        "n_bins": 10,
        "validation": {
            "original": probability_quality(y_validation, validation_probabilities[best_name]),
            "sigmoid": probability_quality(y_validation, calibrated_validation),
        },
        "test": {
            "original": probability_quality(y_test, test_probabilities[best_name]),
            "sigmoid": probability_quality(
                y_test, calibration_candidate.predict_proba(x_test)[:, 1]
            ),
        },
    }
    background_size = min(100, len(x_train))
    explanation_background = x_train.sample(n=background_size, random_state=random_state)
    remaining = x_train.drop(index=explanation_background.index, errors="ignore")
    explanation_background = explanation_background.reset_index(drop=True)
    if remaining.empty:
        remaining = x_train
    explanation_sample = remaining.sample(
        n=min(250, len(remaining)), random_state=random_state
    )
    importance = global_feature_importance(
        fitted[best_name], explanation_background, explanation_sample
    )
    report: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "global_feature_importance": importance,
        "threshold_counts": threshold_counts(y_validation, validation_probabilities[best_name]),
        "test_threshold_counts": threshold_counts(y_test, test_probabilities[best_name]),
        "threshold_data_source": "validation",
        "validation_models": validation_results,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(x_test),
        "churn_rate": round(float(data[TARGET].mean()), 4),
        "models": results,
        "calibration": calibration_report,
    }

    if model_path:
        output = Path(model_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": ARTIFACT_VERSION,
                "pipeline": fitted[best_name],
                "report": report,
                "explanation_background": explanation_background,
                "calibration_candidate": calibration_candidate,
            },
            output,
        )
    if metrics_path:
        output = Path(metrics_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return fitted[best_name], report


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    """Load a persisted model artifact."""
    return joblib.load(model_path)


def risk_details(
    probability: float,
    high_risk_threshold: float = 0.70,
    medium_risk_threshold: float = 0.40,
) -> tuple[str, str]:
    """Translate a churn probability into a risk tier and retention action."""
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Churn probability must be a finite number between 0 and 1.")
    if not 0 < medium_risk_threshold < high_risk_threshold < 1:
        raise ValueError("Risk thresholds must satisfy 0 < medium < high < 1.")
    if probability >= high_risk_threshold:
        return "High", "Contact the customer and offer a retention incentive."
    if probability >= medium_risk_threshold:
        return "Medium", "Review recent support issues and monitor the account."
    return "Low", "No immediate intervention is required."


def predict_customer(
    pipeline: Pipeline,
    customer: dict[str, Any],
    high_risk_threshold: float = 0.70,
    medium_risk_threshold: float = 0.40,
) -> dict[str, Any]:
    """Predict churn risk for one customer and attach an action label."""
    row = pd.DataFrame([customer], columns=FEATURE_COLUMNS)
    probability = float(pipeline.predict_proba(row)[0, 1])
    risk, action = risk_details(
        probability,
        high_risk_threshold=high_risk_threshold,
        medium_risk_threshold=medium_risk_threshold,
    )
    return {"churn_probability": round(probability, 4), "risk_level": risk, "action": action}
