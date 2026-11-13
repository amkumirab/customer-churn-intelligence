import json
from pathlib import Path

import numpy as np

from src.data import FEATURE_COLUMNS, generate_customers
from src.explain import explain_customer
from src.model import ARTIFACT_VERSION, load_artifact, predict_customer, train_and_evaluate


def test_training_returns_metrics_prediction_and_explanations(tmp_path: Path) -> None:
    data = generate_customers(n_rows=500, random_state=17)
    model_path = tmp_path / "churn_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    pipeline, report = train_and_evaluate(
        data, model_path=model_path, metrics_path=metrics_path, random_state=17
    )
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == report

    assert report["best_model"] in {"logistic_regression", "random_forest"}
    assert 0.5 <= report["models"][report["best_model"]]["roc_auc"] <= 1.0
    assert report["artifact_version"] == ARTIFACT_VERSION
    assert len(report["global_feature_importance"]) == 10
    assert report["global_feature_importance"][0]["importance"] > 0
    assert len(report["threshold_counts"]) == 17
    calibration = report["calibration"]
    assert calibration["status"] == "diagnostic_only"
    assert calibration["active_variant"] == "original"
    assert calibration["fit_source"] == "training_out_of_fold"
    assert calibration["folds"] == 5
    for split, row_count in (
        ("validation", report["validation_rows"]), ("test", report["test_rows"])
    ):
        assert set(calibration[split]) == {"original", "sigmoid"}
        for quality in calibration[split].values():
            assert quality["samples"] == row_count
            assert sum(row["count"] for row in quality["bins"]) == row_count
            assert 0 <= quality["brier_score"] <= 1
            assert np.isfinite(quality["log_loss"])

    customer = data.iloc[0][FEATURE_COLUMNS].to_dict()
    result = predict_customer(pipeline, customer)
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}

    artifact = load_artifact(model_path)
    comparison_candidate = artifact["calibration_candidate"]
    candidate_base = comparison_candidate.calibrated_classifiers_[0].estimator
    assert candidate_base is not pipeline
    np.testing.assert_allclose(
        candidate_base.predict_proba(data[FEATURE_COLUMNS]),
        pipeline.predict_proba(data[FEATURE_COLUMNS]),
    )
    calibrated_scores = comparison_candidate.predict_proba(data[FEATURE_COLUMNS])
    assert np.isfinite(calibrated_scores).all()
    assert ((calibrated_scores >= 0) & (calibrated_scores <= 1)).all()
    explanation = explain_customer(
        artifact["pipeline"], customer, artifact["explanation_background"], top_n=5
    )
    assert len(explanation) == 5
    assert explanation["impact"].abs().sum() > 0
    assert set(explanation["direction"]).issubset({"Increases risk", "Decreases risk"})
