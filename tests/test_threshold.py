import pytest

from src.threshold import (
    cost_analysis,
    recommended_threshold,
    threshold_counts,
    threshold_scenario,
)


def test_threshold_counts_match_known_confusion_matrix() -> None:
    counts = threshold_counts(
        [0, 0, 1, 1],
        [0.10, 0.60, 0.40, 0.90],
        thresholds=[0.50],
    )

    assert counts == [
        {
            "threshold": 0.50,
            "true_positive": 1,
            "false_positive": 1,
            "true_negative": 1,
            "false_negative": 1,
            "precision": 0.50,
            "recall": 0.50,
        }
    ]


def test_cost_assumptions_change_the_recommended_threshold() -> None:
    counts = threshold_counts(
        [0, 0, 1, 1],
        [0.10, 0.60, 0.40, 0.90],
        thresholds=[0.30, 0.70],
    )

    retention_focused = recommended_threshold(counts, 10, 100)
    outreach_focused = recommended_threshold(counts, 100, 10)

    assert retention_focused["threshold"] == pytest.approx(0.30)
    assert retention_focused["estimated_cost"] == pytest.approx(30.0)
    assert outreach_focused["threshold"] == pytest.approx(0.70)
    assert outreach_focused["estimated_cost"] == pytest.approx(110.0)


def test_selected_scenario_uses_the_closest_evaluated_threshold() -> None:
    counts = threshold_counts(
        [0, 0, 1, 1],
        [0.10, 0.60, 0.40, 0.90],
        thresholds=[0.30, 0.50, 0.70],
    )

    scenario = threshold_scenario(counts, 0.48, 10, 100)
    analysis = cost_analysis(counts, 10, 100)

    assert scenario["threshold"] == pytest.approx(0.50)
    assert len(analysis) == 3


def test_invalid_costs_and_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="Probabilities"):
        threshold_counts([0, 1], [0.20, 1.20])
    with pytest.raises(ValueError, match="cannot be negative"):
        cost_analysis([{"threshold": 0.50}], -1, 100)
