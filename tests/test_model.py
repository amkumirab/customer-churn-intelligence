from src.data import FEATURE_COLUMNS, generate_customers
from src.model import predict_customer, train_and_evaluate


def test_training_returns_metrics_and_prediction() -> None:
    data = generate_customers(n_rows=500, random_state=17)
    pipeline, report = train_and_evaluate(data, random_state=17)

    assert report["best_model"] in {"logistic_regression", "random_forest"}
    assert 0.5 <= report["models"][report["best_model"]]["roc_auc"] <= 1.0

    customer = data.iloc[0][FEATURE_COLUMNS].to_dict()
    result = predict_customer(pipeline, customer)
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}
