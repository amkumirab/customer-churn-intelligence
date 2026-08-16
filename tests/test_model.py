from pathlib import Path

from src.data import FEATURE_COLUMNS, generate_customers
from src.explain import explain_customer
from src.model import ARTIFACT_VERSION, load_artifact, predict_customer, train_and_evaluate


def test_training_returns_metrics_prediction_and_explanations(tmp_path: Path) -> None:
    data = generate_customers(n_rows=500, random_state=17)
    model_path = tmp_path / "churn_model.joblib"
    pipeline, report = train_and_evaluate(data, model_path=model_path, random_state=17)

    assert report["best_model"] in {"logistic_regression", "random_forest"}
    assert 0.5 <= report["models"][report["best_model"]]["roc_auc"] <= 1.0
    assert report["artifact_version"] == ARTIFACT_VERSION
    assert len(report["global_feature_importance"]) == 10
    assert report["global_feature_importance"][0]["importance"] > 0
    assert len(report["threshold_counts"]) == 17

    customer = data.iloc[0][FEATURE_COLUMNS].to_dict()
    result = predict_customer(pipeline, customer)
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}

    artifact = load_artifact(model_path)
    explanation = explain_customer(
        artifact["pipeline"], customer, artifact["explanation_background"], top_n=5
    )
    assert len(explanation) == 5
    assert explanation["impact"].abs().sum() > 0
    assert set(explanation["direction"]).issubset({"Increases risk", "Decreases risk"})
