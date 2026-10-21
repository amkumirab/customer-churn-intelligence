from sklearn.ensemble import RandomForestClassifier

from src.data import FEATURE_COLUMNS, load_kaggle_telco
from src.explain import explain_customer, global_feature_importance
from src.model import build_pipeline


def test_forest_explanations_work_with_a_small_training_sample():
    data = load_kaggle_telco("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    features = data[FEATURE_COLUMNS]
    pipeline = build_pipeline(
        RandomForestClassifier(n_estimators=30, min_samples_leaf=4, random_state=42)
    )
    pipeline.fit(features.head(500), data.churn.head(500))
    importance = global_feature_importance(pipeline, features.head(50), features.tail(20))
    explanation = explain_customer(pipeline, features.iloc[-1].to_dict(), features.head(50))
    assert len(importance) == 10
    assert len(explanation) == 6
    assert explanation.impact.abs().sum() > 0
