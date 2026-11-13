import json
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


def test_workspace_has_focused_views_and_a_sample_data_path(dashboard):
    assert [tab.label for tab in dashboard.tabs] == [
        "Customer review", "Campaign workspace", "Model evaluation",
    ]
    next(b for b in dashboard.button if b.label == "Use sample customers").click().run(timeout=60)
    assert any(m.label == "Customers scored" and m.value == "3" for m in dashboard.metric)
    assert any(m.label == "Matching customers" and m.value == "3" for m in dashboard.metric)
    assert any(m.label == "Planned contacts" for m in dashboard.metric)
    assert not dashboard.exception


def test_campaign_respects_zero_budget_after_rerun(dashboard):
    next(b for b in dashboard.button if b.label == "Use sample customers").click().run(timeout=60)
    dashboard.number_input(key="campaign_budget").set_value(0.0).run(timeout=60)
    contacts = next(m for m in dashboard.metric if m.label == "Planned contacts")
    assert contacts.value == "0"
    assert any("No customers selected" in info.value for info in dashboard.info)
    assert not dashboard.exception


def test_empty_filter_selection_is_not_silently_treated_as_all(dashboard):
    next(b for b in dashboard.button if b.label == "Use sample customers").click().run(timeout=60)
    dashboard.multiselect(key="cohort_risk_filter").set_value([]).run(timeout=60)
    assert next(m for m in dashboard.metric if m.label == "Matching customers").value == "0"
    assert next(m for m in dashboard.metric if m.label == "Planned contacts").value == "0"
    assert not dashboard.exception


def test_duplicate_upload_is_rejected_for_campaign_preparation(monkeypatch, dashboard):
    customers = pd.read_csv("data/example_customers.csv")
    customers.insert(0, "customer_id", ["001", "001", "002"])
    payload = customers.to_csv(index=False).encode()
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: BytesIO(payload))
    dashboard.run(timeout=60)
    assert any("unique" in error.value for error in dashboard.error)
    assert not dashboard.exception


def test_model_evaluation_distinguishes_validation_and_test(dashboard):
    headings = [heading.value for heading in dashboard.subheader]
    assert "Model comparison" in headings
    assert "Final-test confusion matrix" in headings
    assert "Validation threshold trade-offs" in headings
    assert any("not calibrated" in warning.value for warning in dashboard.warning)
    assert not dashboard.exception


def test_probability_quality_comparison_defaults_to_validation(dashboard, dashboard_artifact):
    assert dashboard.selectbox(key="calibration_split").value == "Validation"
    comparison = next(
        table.value for table in dashboard.dataframe if "brier_score" in table.value.columns
    )
    assert len(comparison) == 2
    assert comparison["samples"].tolist() == [dashboard_artifact["report"]["validation_rows"]] * 2
    assert any("does not replace" in info.value for info in dashboard.info)
    assert not dashboard.exception


def test_diagnostic_split_does_not_change_live_prediction(dashboard, dashboard_artifact):
    next(b for b in dashboard.button if b.label == "Calculate churn risk").click().run(timeout=60)
    before = next(m.value for m in dashboard.metric if m.label == "Predicted churn probability")
    dashboard.selectbox(key="calibration_split").set_value("Final test").run(timeout=60)
    after = next(m.value for m in dashboard.metric if m.label == "Predicted churn probability")
    assert before == after
    comparison = next(
        table.value for table in dashboard.dataframe if "brier_score" in table.value.columns
    )
    assert comparison["samples"].tolist() == [dashboard_artifact["report"]["test_rows"]] * 2
    assert not dashboard.exception


def test_downloadable_calibration_report_matches_the_artifact(
    monkeypatch, dashboard, dashboard_artifact
):
    captured = {}

    def capture_download(label, data, **kwargs):
        if kwargs.get("key") == "download_calibration_report":
            captured["report"] = json.loads(data)
        return False

    monkeypatch.setattr("streamlit.download_button", capture_download)
    dashboard.run(timeout=60)
    assert captured["report"] == dashboard_artifact["report"]["calibration"]
    assert not dashboard.exception
