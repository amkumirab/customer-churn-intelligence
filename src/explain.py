"""SHAP-based global and local explanations for trained churn pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def _dense(values: Any) -> np.ndarray:
    """Convert transformed feature matrices to dense arrays for SHAP."""
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def _feature_identity(transformed_name: str) -> tuple[str, str]:
    """Map a transformed feature back to its raw field and a readable label."""
    clean_name = transformed_name.split("__", maxsplit=1)[-1]
    for feature in CATEGORICAL_FEATURES:
        prefix = f"{feature}_"
        if clean_name.startswith(prefix):
            category = clean_name[len(prefix) :]
            label = f"{feature.replace('_', ' ').title()} = {category}"
            return feature, label
    return clean_name, clean_name.replace("_", " ").title()


def _build_explainer(pipeline: Pipeline, background: pd.DataFrame) -> tuple[Any, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())
    transformed_background = _dense(preprocessor.transform(background[FEATURE_COLUMNS]))

    if isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(
            classifier, transformed_background, feature_names=feature_names
        )
    elif isinstance(classifier, RandomForestClassifier):
        explainer = shap.TreeExplainer(
            classifier, transformed_background, feature_names=feature_names
        )
    else:
        explainer = shap.Explainer(
            classifier, transformed_background, feature_names=feature_names
        )
    return explainer, feature_names


def _positive_class_values(explanation: Any) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def explain_customer(
    pipeline: Pipeline,
    customer: dict[str, Any],
    background: pd.DataFrame,
    top_n: int = 6,
) -> pd.DataFrame:
    """Return the strongest raw-feature contributions for one prediction."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    row = pd.DataFrame([customer], columns=FEATURE_COLUMNS)
    preprocessor = pipeline.named_steps["preprocessor"]
    explainer, feature_names = _build_explainer(pipeline, background)
    explanation = explainer(_dense(preprocessor.transform(row)))
    shap_values = _positive_class_values(explanation)[0]

    contributions: dict[str, float] = {feature: 0.0 for feature in FEATURE_COLUMNS}
    for transformed_name, shap_value in zip(feature_names, shap_values, strict=True):
        raw_feature, _ = _feature_identity(transformed_name)
        contributions[raw_feature] += float(shap_value)

    rows = [
        {
            "feature": feature.replace("_", " ").title(),
            "value": str(customer[feature]),
            "impact": round(impact, 4),
            "direction": "Increases risk" if impact >= 0 else "Decreases risk",
        }
        for feature, impact in contributions.items()
    ]
    result = pd.DataFrame(rows)
    result["absolute_impact"] = result["impact"].abs()
    return (
        result.sort_values("absolute_impact", ascending=False)
        .head(top_n)
        .drop(columns="absolute_impact")
        .reset_index(drop=True)
    )


def global_feature_importance(
    pipeline: Pipeline,
    background: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Aggregate mean absolute SHAP values back to the original input fields."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    preprocessor = pipeline.named_steps["preprocessor"]
    explainer, feature_names = _build_explainer(pipeline, background)
    transformed = _dense(preprocessor.transform(evaluation_data[FEATURE_COLUMNS]))
    shap_values = np.abs(_positive_class_values(explainer(transformed)))

    grouped: dict[str, float] = {feature: 0.0 for feature in FEATURE_COLUMNS}
    for index, transformed_name in enumerate(feature_names):
        raw_feature, _ = _feature_identity(transformed_name)
        grouped[raw_feature] += float(shap_values[:, index].mean())

    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        {
            "feature": feature.replace("_", " ").title(),
            "importance": round(importance, 4),
        }
        for feature, importance in ranked
    ]
