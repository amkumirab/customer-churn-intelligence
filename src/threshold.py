"""Decision-threshold evaluation and retention cost analysis."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.10, 0.91, 0.05))


def threshold_counts(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Calculate confusion-matrix counts across candidate thresholds."""
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Labels and probabilities must have the same non-zero length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0 < threshold < 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        predicted = scores >= threshold
        true_positive = int((predicted & (labels == 1)).sum())
        false_positive = int((predicted & (labels == 0)).sum())
        true_negative = int((~predicted & (labels == 0)).sum())
        false_negative = int((~predicted & (labels == 1)).sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        rows.append(
            {
                "threshold": round(threshold, 2),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "precision": round(
                    true_positive / precision_denominator if precision_denominator else 0.0,
                    4,
                ),
                "recall": round(
                    true_positive / recall_denominator if recall_denominator else 0.0,
                    4,
                ),
            }
        )
    return rows


def cost_analysis(
    counts: list[dict[str, Any]],
    intervention_cost: float,
    missed_churn_cost: float,
) -> pd.DataFrame:
    """Estimate policy cost for each threshold and rank the lowest-cost option."""
    if intervention_cost < 0 or missed_churn_cost < 0:
        raise ValueError("Cost assumptions cannot be negative.")
    if not counts:
        raise ValueError("Threshold counts cannot be empty.")

    analysis = pd.DataFrame(counts).copy()
    analysis["customers_contacted"] = (
        analysis["true_positive"] + analysis["false_positive"]
    )
    analysis["estimated_cost"] = (
        analysis["customers_contacted"] * float(intervention_cost)
        + analysis["false_negative"] * float(missed_churn_cost)
    ).round(2)
    return analysis.sort_values(
        ["estimated_cost", "recall", "threshold"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def recommended_threshold(
    counts: list[dict[str, Any]],
    intervention_cost: float,
    missed_churn_cost: float,
) -> dict[str, Any]:
    """Return the lowest-cost threshold scenario."""
    return cost_analysis(counts, intervention_cost, missed_churn_cost).iloc[0].to_dict()


def threshold_scenario(
    counts: list[dict[str, Any]],
    threshold: float,
    intervention_cost: float,
    missed_churn_cost: float,
) -> dict[str, Any]:
    """Return the closest evaluated scenario for a selected threshold."""
    analysis = cost_analysis(counts, intervention_cost, missed_churn_cost)
    index = (analysis["threshold"] - float(threshold)).abs().idxmin()
    return analysis.loc[index].to_dict()
