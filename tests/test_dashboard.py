from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import model
from src.data import generate_customers
from src.threshold import threshold_scenario


@pytest.fixture(scope="module")
def dashboard_artifact(tmp_path_factory):
    path = tmp_path_factory.mktemp("dashboard") / "model.joblib"
    model.train_and_evaluate(generate_customers(500, random_state=17), model_path=path)
    return model.load_artifact(path)


@pytest.fixture
def dashboard(monkeypatch, dashboard_artifact):
    # Isolate model-file IO while exercising the real dashboard and trained pipeline.
    monkeypatch.setattr(model, "load_artifact", lambda path: dashboard_artifact)
    monkeypatch.setattr(model, "train_and_evaluate", lambda *args, **kwargs: None)
    script = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(script)).run(timeout=60)
    assert not app.exception
    return app


def test_recommended_threshold_can_be_applied_and_recall_matches_policy(
    dashboard, dashboard_artifact
):
    use_recommendation = next(
        button for button in dashboard.sidebar.button if button.label == "Use recommended threshold"
    )
    use_recommendation.click().run(timeout=60)
    metric = next(m for m in dashboard.sidebar.metric if m.label == "Recommended threshold")
    assert dashboard.sidebar.slider[0].value == pytest.approx(float(metric.value.strip("%")) / 100)
    policy = threshold_scenario(
        dashboard_artifact["report"]["test_threshold_counts"],
        dashboard.sidebar.slider[0].value, 25, 300,
    )
    recall = next(m for m in dashboard.metric if m.label == "Test recall (selected policy)")
    assert float(recall.value) == pytest.approx(policy["recall"], abs=0.001)
    assert not dashboard.exception


def test_prediction_persists_when_the_threshold_changes(dashboard):
    next(b for b in dashboard.button if b.label == "Calculate churn risk").click().run(timeout=60)
    dashboard.sidebar.slider[0].set_value(0.10).run(timeout=60)
    assert any(m.label == "Predicted churn probability" for m in dashboard.metric)
    assert not dashboard.exception


@pytest.mark.parametrize("content", [b"", b"invalid,columns\n1,2\n"])
def test_invalid_uploads_show_errors_without_crashing(monkeypatch, dashboard, content):
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: BytesIO(content))
    dashboard.run(timeout=60)
    assert dashboard.error
    assert not dashboard.exception


def test_valid_upload_can_be_scored(monkeypatch, dashboard):
    payload = pd.read_csv("data/example_customers.csv").to_csv(index=False).encode()
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: BytesIO(payload))
    dashboard.run(timeout=60)
    assert any(m.label == "Customers scored" and m.value == "3" for m in dashboard.metric)
    assert not dashboard.exception
