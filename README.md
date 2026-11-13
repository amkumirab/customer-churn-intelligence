# Customer Churn Intelligence

[![Tests](https://github.com/amkumirab/customer-churn-intelligence/actions/workflows/tests.yml/badge.svg)](https://github.com/amkumirab/customer-churn-intelligence/actions)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine-learning portfolio project for telecom churn review and retention
planning. Compare classifiers, explain customer scores, inspect cohort segments, and prepare
a budget-constrained outreach scenario with a downloadable decision bundle.

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
- Training-only sigmoid calibration diagnostics with Brier/log loss and reliability diagrams
- Global and per-customer explanations using SHAP values
- Batch CSV scoring with validation, risk ranking, and downloadable results
- Cost-aware decision thresholds with Precision and Recall trade-off estimates
- Budget- and capacity-constrained campaign planning with deterministic contact priorities
- Operational cohort segmentation, risk/contract filters, and score-weighted revenue exposure
- In-memory decision bundles with protected CSVs, assumptions, and provenance fingerprints
- Task-oriented Streamlit views for customer review, campaigns, and model evaluation
- Automated tests with GitHub Actions and optional Docker deployment

## Architecture

```text
Kaggle CSV or synthetic fallback
        |
        v
Train / validation / test -> train-only preprocessing -> validation model selection
                                                               |
                                                     Versioned model artifact
                                                               |
                          +------------------------------------+---------------------+
                          |                                    |                     |
                    Customer review                      Cohort scoring        Model evaluation
                    + SHAP factors                             |
                                                    Filters + segment analysis
                                                               |
                                                    Budget / capacity allocation
                                                               |
                                                    CSV + ZIP decision bundle
```

## Baseline results

The reproducible default run uses 7,043 Kaggle records with stratified training (60%),
validation (15%), and final test (25%) splits. Models and cost thresholds are selected using
validation data only. Final test labels cannot influence those decisions.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.846** | 0.748 | 0.516 | **0.803** | 0.628 |
| Random Forest | 0.843 | **0.768** | **0.544** | 0.765 | **0.636** |

The best model is selected by validation ROC-AUC. The table reports final test metrics at a
fixed threshold of 50%; the dashboard separately reports test recall at the selected policy.
These are portfolio benchmarks, not production business performance.

## Probability quality and calibration

The **Model evaluation** tab compares the selected original model with a separate
sigmoid-calibrated candidate. It reports Brier score, log loss, observed prevalence, sample
counts, and reliability points for validation and final test separately. Download the aggregate
comparison as `calibration_report.json`; it contains no individual customer records.

Calibration uses five seeded stratified folds on training rows only, with the entire
preprocessing pipeline refitted in each fold. Out-of-fold predictions fit the sigmoid mapping;
the candidate's base estimator is then refitted on all training rows. Validation and final-test
labels never fit the calibrator. For very small training populations, folds are reduced to the
minority-class count, with at least two rows per class required.

This release is **diagnostic only**. The original pipeline remains active for single-customer
predictions, campaign scoring, thresholds, and SHAP explanations. The candidate is stored under
a separate artifact key rather than silently promoted using final-test results. Artifact schema
v5 triggers regeneration of older local artifacts on the next dashboard launch.

One default Telco run (seed 42, scikit-learn 1.9.0, 1,761 final-test rows) produced:

| Variant | Final-test Brier score | Final-test log loss |
|---|---:|---:|
| Original Logistic Regression (active) | 0.16654 | 0.49216 |
| Sigmoid candidate (comparison only) | 0.13660 | 0.41589 |

Lower losses indicate better overall probability quality, not necessarily calibration alone.
The diagram also exposes counts in ten fixed-width bins: empty bins remain null and are not
plotted, and small bins are noisy. These results do not establish future performance, causal
retention impact, or a reason to promote a model using test data. See
[calibration design and verification](docs/CALIBRATION.md) and the
[official calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

## Explainable predictions

The dashboard makes model output easier to audit and communicate:

- **Global importance** ranks the original customer fields by mean absolute SHAP value.
- **Local explanations** show which factors increased or reduced one customer's predicted risk.
- One-hot encoded values are grouped back into readable business fields.
- A versioned model artifact stores a representative background sample for reproducibility.

Linear explanations use the stored training background; forest explanations use training-path
coverage. SHAP's additivity checks stay enabled for both model families.

SHAP values explain model behavior and associations; they do not prove that a feature causes
customer churn.

## Batch predictions

The dashboard can score up to 10,000 customers in one upload:

- accepts the included project template or the original Kaggle Telco column names
- validates required fields and returns readable data-quality errors
- preserves `customer_id` or Kaggle `customerID` values when available
- preserves leading zeros and literal identifiers such as `NA`
- requires unique, nonblank customer IDs for the campaign workspace
- rejects empty files, non-finite charges, negative amounts, and unsupported categories
- ranks customers by churn probability and assigns Low, Medium, or High risk
- summarizes the high- and medium-risk customer counts
- exports a retention-ready CSV with recommended actions

CSV exports prefix formula-like text with an apostrophe so spreadsheet applications treat
customer identifiers as text. Numeric prediction columns remain numeric.

Download the input template directly from the dashboard or use
[`data/example_customers.csv`](data/example_customers.csv) as a schema reference.

## Retention campaign workspace

Open **Campaign workspace**, then upload a CSV or select **Use sample customers**.
Use the risk-level and contract filters to define the population. Empty filter selections
match no customers; they never silently expand the campaign to everyone.

The workspace provides:

- cohort counts, imputation warnings, and score-weighted monthly revenue exposure
- segment comparisons by contract, internet service, payment method, or tenure band
- a uniform outreach cost from the policy sidebar, a monetary budget, and a contact capacity
- ranking by highest churn score or highest scenario net value, with customer ID as a tie-breaker
- an optional exclusion of contacts with non-positive scenario net value
- budget sensitivity and exportable contact priorities

Only customers at or above the active high-risk threshold are eligible for outreach. Budget
arithmetic uses integer cents, including small amounts such as $0.30 / $0.10. A zero-cost
scenario remains capacity-limited. Scores retain full precision for eligibility; formatting
must not promote a just-below-threshold customer into the campaign.

Financial scenarios use explicit, editable assumptions:

```text
Scenario retained customers = churn score x assumed retention success rate
Scenario retained revenue   = retained customers x monthly charges x horizon months
Scenario net value          = retained revenue - outreach cost
```

For example, a score of 0.80, $100 monthly charges, a 20% assumed success rate, a six-month
horizon, and $25 outreach cost produce $96 scenario retained revenue and $71 net value.
This is arithmetic under assumptions, not evidence that outreach will cause that result.
Class-balanced models are not calibrated probability forecasts. These values are neither
profit nor customer lifetime value; margins, consent, contactability, and treatment effects
are not modeled. No customer messages or campaigns are executed.

**Download decision bundle** creates a ZIP in memory containing:

| File | Contents |
|---|---|
| `scored_customers.csv` | Filtered cohort scores and operational segmentation fields |
| `contact_list.csv` | Ranked contacts, outreach cost, and per-contact scenario values |
| `contract_segments.csv` | Reconciled contract-level summaries |
| `budget_scenarios.csv` | Zero, half-budget, and selected-budget allocations |
| `manifest.json` | Filters, thresholds, configuration, totals, model report identity, and cohort hash |
| `README.txt` | Definitions, safety notes, and modeling limitations |

The manifest's cohort fingerprint hashes the exported, spreadsheet-protected cohort sorted
by literal customer ID. The model report fingerprint identifies report contents, not model
weights. Bundles exclude raw demographic profiles and uploaded churn labels, but still contain
customer identifiers and financial fields. Uploaded records are not persisted by this code;
downloaded files require appropriate handling. See [workspace design](docs/RETENTION_WORKSPACE.md).

## Decision policy and cost assumptions

The dashboard sidebar supports scenario-based threshold selection:

- set the estimated cost of contacting one customer
- set the estimated cost of missing one customer who churns
- compare the selected policy with the lowest-cost evaluated threshold
- apply any recommended threshold between 10% and 90% with one click
- review the expected Precision, Recall, outreach volume, and evaluation cost
- apply the selected risk thresholds to both single and batch predictions

The headline test recall follows the selected threshold. Validation cost estimates are labeled
separately, and the last submitted customer profile stays visible when the policy changes.

The cost estimates use validation counts and simplified cost assumptions. They are intended
for comparing policies, not for forecasting exact financial outcomes.

## Quick start

Use Python 3.12 or newer. SHAP 0.52 requires Python 3.12; the test runner and Docker image
also use Python 3.12.

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

The suite covers campaign allocation, scoring-to-feature alignment, safe bundles, and real
Streamlit AppTest interactions in addition to training and CSV regressions. Browser visual
checks, Docker checks, and remote runs remain separate gates; see the
[release checklist](docs/RELEASE_CHECKLIST.md).

## Container build

```bash
docker build -t customer-churn-intelligence .
docker run --rm -p 8501:8501 customer-churn-intelligence
```

The build excludes virtual environments, local model artifacts, credentials, and scratch
files through `.dockerignore`. The model is trained from the included dataset on first launch.
Runtime requirements include minimum patched versions of Pillow, certifi, and urllib3;
development requirements use pytest 9.0.3 or newer.

## Repository structure

```text
app/                 Streamlit user interface
data/                Kaggle dataset, attribution, and schema example
src/data.py           Kaggle cleaner and synthetic fallback generator
src/batch.py          CSV validation, batch scoring, summaries, and export data
src/campaign.py       Budget/capacity allocation and sensitivity scenarios
src/calibration.py    Training-only calibration and probability-quality diagnostics
src/cohort.py         Identity-safe scoring, filtering, and segment aggregation
src/explain.py        SHAP global and local explanation utilities
src/model.py          Preprocessing, training, evaluation and prediction
src/reporting.py      In-memory ZIP exports and provenance manifest
src/threshold.py      Threshold evaluation and retention cost scenarios
src/train.py          Training command-line interface
tests/                Automated tests
docs/                 Workspace design and release verification checklist
.github/workflows/    Continuous integration
```

## Important modeling choices

- Training, validation, and test splits are stratified because churn classes can be imbalanced.
- Preprocessing is fitted only on training data to prevent data leakage.
- Models use balanced class weights so minority-class errors matter during training.
- Validation ROC-AUC selects the best model; final test data is reserved for reporting.

## Suggested next improvements

- Add drift monitoring and model versioning
- Deploy the dashboard to Streamlit Community Cloud
- Define validation-only promotion criteria before activating a calibrated candidate
- Evaluate actual retention treatment effects before attempting business deployment

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
