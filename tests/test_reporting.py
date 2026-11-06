import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.batch import export_predictions_csv
from src.campaign import CampaignConfig
from src.cohort import score_cohort
from src.reporting import COHORT_EXPORT_COLUMNS, build_decision_bundle


class ChargeBasedPipeline:
    def predict_proba(self, customers):
        positive = customers["monthly_charges"].to_numpy() / 200
        return np.column_stack([1 - positive, positive])


def test_decision_bundle_is_safe_and_carries_reproducible_provenance():
    customers = pd.read_csv("data/example_customers.csv")
    customers.insert(0, "customer_id", ["=1+1", "001", "@SUM(1)"])
    cohort = score_cohort(ChargeBasedPipeline(), customers, 0.4, 0.1)
    config = CampaignConfig(budget=100, minimum_probability=0.4)
    report = {"best_model": "logistic_regression", "artifact_version": 4, "test_rows": 100}
    filters = {
        "risk_levels": ["High", "Medium", "Low"],
        "contracts": ["Month-to-month", "One year", "Two year"],
    }
    payload = build_decision_bundle(cohort, config, report, 0.4, 0.1, population_filters=filters)
    with ZipFile(BytesIO(payload)) as bundle:
        assert set(bundle.namelist()) == {
            "manifest.json", "scored_customers.csv", "contact_list.csv",
            "contract_segments.csv", "budget_scenarios.csv", "README.txt",
        }
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["schema_version"] == 1
        assert manifest["model"]["name"] == "logistic_regression"
        assert manifest["policy"] == {"high_risk_threshold": 0.4, "medium_risk_threshold": 0.1}
        assert manifest["assumptions"]["success_rate"] == 0.2
        assert manifest["cohort_rows"] == 3
        assert manifest["population_filters"] == filters
        assert manifest["summary"]["contacts"] == 1
        canonical = export_predictions_csv(
            cohort[COHORT_EXPORT_COLUMNS].sort_values("customer_id").reset_index(drop=True)
        )
        assert manifest["cohort_sha256"] == hashlib.sha256(canonical).hexdigest()
        exported = pd.read_csv(BytesIO(bundle.read("scored_customers.csv")), dtype="string")
        assert set(exported["customer_id"]) == {"'=1+1", "001", "'@SUM(1)"}
        private_columns = {"gender", "churn", "has_partner", "total_charges"}
        assert not private_columns.intersection(exported.columns)
        assert "not calibrated" in bundle.read("README.txt").decode()
        assert "gender" not in bundle.read("contact_list.csv").decode()


def test_empty_campaign_exports_headers_and_json_null_roi():
    raw = pd.read_csv("data/example_customers.csv")
    cohort = score_cohort(ChargeBasedPipeline(), raw)
    config = CampaignConfig(budget=0)
    payload = build_decision_bundle(cohort.iloc[:0], config, {}, 0.7, 0.4)
    with ZipFile(BytesIO(payload)) as bundle:
        assert pd.read_csv(BytesIO(bundle.read("contact_list.csv"))).empty
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["summary"]["scenario_roi"] is None
        assert manifest["cohort_rows"] == 0


def test_fingerprint_ignores_row_order_but_detects_changed_scores():
    cohort = score_cohort(ChargeBasedPipeline(), pd.read_csv("data/example_customers.csv"))
    config = CampaignConfig()

    def fingerprint(data):
        with ZipFile(BytesIO(build_decision_bundle(data, config, {}, 0.7, 0.4))) as bundle:
            return json.loads(bundle.read("manifest.json"))["cohort_sha256"]

    first = fingerprint(cohort)
    assert fingerprint(cohort.iloc[::-1]) == first
    cohort.loc[0, "churn_probability"] = 0.99
    assert fingerprint(cohort) != first
