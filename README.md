# Customer Churn Intelligence

[![Tests](https://github.com/amkumirab/customer-churn-intelligence/actions/workflows/tests.yml/badge.svg)](https://github.com/amkumirab/customer-churn-intelligence/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine-learning portfolio project that predicts telecom customer churn,
compares classification models, and turns probability scores into simple retention actions.

The project trains on the public **IBM Telco Customer Churn** dataset downloaded from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 fictional
telecom customers, 19 model features, and a binary churn target.

> This repository is a technical portfolio demonstration, not a production decision system.

## What this project demonstrates

- Cleaning and validation of a real-world tabular dataset
- A reproducible synthetic-data fallback for offline development
- Leakage-safe preprocessing using a scikit-learn `Pipeline`
- Missing-value handling, scaling, and one-hot encoding
- Comparison of Logistic Regression and Random Forest
- Evaluation with ROC-AUC, accuracy, precision, recall, and F1
- Global and per-customer explanations using SHAP values
- Batch CSV scoring with validation, risk ranking, and downloadable results
- Interactive Streamlit dashboard with risk-based recommendations
- Automated tests with GitHub Actions and optional Docker deployment

## Architecture

```text
Kaggle CSV or synthetic fallback
        |
        v
Train/test split -> preprocessing -> model comparison -> best model
                                                       |
                                      +----------------+----------------+
                                      |                                 |
                                      v                                 v
                              SHAP explanations                 Streamlit dashboard
```

## Baseline results

The reproducible default run uses all 7,043 Kaggle records and a stratified 25% test split.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.846** | 0.750 | 0.519 | **0.797** | 0.628 |
| Random Forest | 0.842 | **0.771** | **0.549** | 0.769 | **0.641** |

Logistic Regression is selected by test ROC-AUC. Recall is highlighted because failing to flag
a customer who actually churns can be expensive. Results are a portfolio baseline and should
not be interpreted as production business performance.

## Explainable predictions

The dashboard makes model output easier to audit and communicate:

- **Global importance** ranks the original customer fields by mean absolute SHAP value.
- **Local explanations** show which factors increased or reduced one customer's predicted risk.
- One-hot encoded values are grouped back into readable business fields.
- A versioned model artifact stores a representative background sample for reproducibility.

SHAP values explain model behavior and associations; they do not prove that a feature causes
customer churn.

## Batch predictions

The dashboard can score up to 10,000 customers in one upload:

- accepts the included project template or the original Kaggle Telco column names
- validates required fields and returns readable data-quality errors
- preserves `customer_id` or Kaggle `customerID` values when available
- ranks customers by churn probability and assigns Low, Medium, or High risk
- summarizes the high- and medium-risk customer counts
- exports a retention-ready CSV with recommended actions

Download the input template directly from the dashboard or use
[`data/example_customers.csv`](data/example_customers.csv) as a schema reference.

## Quick start

```bash
git clone https://github.com/amkumirab/customer-churn-intelligence.git
cd customer-churn-intelligence
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first launch automatically cleans the included Kaggle dataset and trains the model.

## Train from the command line

Train with the included Kaggle dataset:

```bash
python -m src.train
```

Or use a compatible CSV file:

```bash
python -m src.train --data data/your_customers.csv
```

Required fields are shown in [`data/example_customers.csv`](data/example_customers.csv).

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## Repository structure

```text
app/                 Streamlit user interface
data/                Kaggle dataset, attribution, and schema example
src/data.py           Kaggle cleaner and synthetic fallback generator
src/batch.py          CSV validation, batch scoring, summaries, and export data
src/explain.py        SHAP global and local explanation utilities
src/model.py          Preprocessing, training, evaluation and prediction
src/train.py          Training command-line interface
tests/                Automated tests
.github/workflows/    Continuous integration
```

## Important modeling choices

- The train/test split is stratified because churn classes can be imbalanced.
- Preprocessing is fitted only on training data to prevent data leakage.
- Models use balanced class weights so minority-class errors matter during training.
- ROC-AUC selects the best model; recall is also reported because missed churners are costly.

## Suggested next improvements

- Tune the decision threshold using retention campaign costs
- Add drift monitoring and model versioning
- Deploy the dashboard to Streamlit Community Cloud
- Add probability calibration and cost-sensitive threshold selection

## Dataset

- **Name:** Telco Customer Churn
- **Source:** [Kaggle — BlastChar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Original source:** IBM Sample Data Sets
- **Size:** 7,043 rows and 21 raw columns
- **Kaggle license label:** Data files © Original Authors

The code license below does not replace or modify the dataset owner's rights. See
[`data/README.md`](data/README.md) for attribution and usage notes.

## License

Project code: MIT. Dataset: © original authors, distributed with attribution to the Kaggle and
IBM sources above.
