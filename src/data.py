"""Load the Kaggle IBM Telco dataset or generate reproducible fallback data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "paperless_billing",
    "payment_method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

KAGGLE_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_COLUMNS = {
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "has_partner",
    "Dependents": "has_dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract_type",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": TARGET,
}


def load_kaggle_telco(path: str | Path) -> pd.DataFrame:
    """Clean and standardize the Kaggle IBM Telco Customer Churn CSV."""
    raw = pd.read_csv(path)
    missing = set(KAGGLE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Kaggle dataset is missing columns: {sorted(missing)}")

    data = raw[list(KAGGLE_COLUMNS)].rename(columns=KAGGLE_COLUMNS).copy()
    data["senior_citizen"] = data["senior_citizen"].map({1: "Yes", 0: "No"})
    data["total_charges"] = pd.to_numeric(data["total_charges"], errors="coerce")
    data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0})

    if data[TARGET].isna().any():
        raise ValueError("Churn contains labels other than Yes/No")
    return data[FEATURE_COLUMNS + [TARGET]]


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load either the original Kaggle schema or the standardized project schema."""
    preview = pd.read_csv(path, nrows=2)
    if set(KAGGLE_COLUMNS).issubset(preview.columns):
        return load_kaggle_telco(path)

    data = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return data[FEATURE_COLUMNS + [TARGET]]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_customers(n_rows: int = 5_000, random_state: int = 42) -> pd.DataFrame:
    """Return fallback synthetic records with the same schema as cleaned Kaggle data."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100")

    rng = np.random.default_rng(random_state)
    yes_no = lambda probability: rng.choice(  # noqa: E731
        ["Yes", "No"], size=n_rows, p=[probability, 1 - probability]
    )
    tenure = rng.integers(1, 73, size=n_rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.35, 0.52, 0.13])
    payment = rng.choice(
        [
            "Electronic check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
            "Mailed check",
        ],
        size=n_rows,
        p=[0.34, 0.24, 0.24, 0.18],
    )
    paperless = yes_no(0.62)
    senior = yes_no(0.16)
    partner = yes_no(0.48)
    tech_support = yes_no(0.30)
    online_security = yes_no(0.29)

    monthly = np.clip(
        20
        + (internet == "DSL") * 25
        + (internet == "Fiber optic") * 55
        + rng.normal(0, 8, size=n_rows),
        18,
        125,
    ).round(2)
    total = np.clip(monthly * tenure + rng.normal(0, 55, size=n_rows), 0, None).round(2)

    log_odds = (
        -1.3
        - 0.032 * tenure
        + 0.020 * (monthly - 65)
        + 1.05 * (contract == "Month-to-month")
        - 0.72 * (contract == "Two year")
        + 0.46 * (internet == "Fiber optic")
        + 0.42 * (payment == "Electronic check")
        + 0.24 * (paperless == "Yes")
        + 0.25 * (senior == "Yes")
        - 0.24 * (partner == "Yes")
        - 0.35 * (tech_support == "Yes")
        - 0.28 * (online_security == "Yes")
        + rng.normal(0, 0.35, size=n_rows)
    )
    churn = rng.binomial(1, _sigmoid(log_odds))

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "gender": rng.choice(["Female", "Male"], size=n_rows),
            "senior_citizen": senior,
            "has_partner": partner,
            "has_dependents": yes_no(0.30),
            "phone_service": yes_no(0.90),
            "multiple_lines": yes_no(0.42),
            "internet_service": internet,
            "online_security": online_security,
            "online_backup": yes_no(0.34),
            "device_protection": yes_no(0.34),
            "tech_support": tech_support,
            "streaming_tv": yes_no(0.38),
            "streaming_movies": yes_no(0.39),
            "contract_type": contract,
            "paperless_billing": paperless,
            "payment_method": payment,
            TARGET: churn,
        }
    )
