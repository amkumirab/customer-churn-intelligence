# Probability quality and calibration diagnostics

## Objective and scope

Compare the selected original model with a sigmoid-calibrated candidate. Add Brier score,
log loss, and fixed-bin reliability diagrams to Model evaluation. This is a diagnostic-only
comparison: the candidate never automatically replaces the active model, thresholds, SHAP
explainer, or campaign scores. Improvement is measured, not assumed.

Use the existing Python/scikit-learn/Streamlit stack without new dependencies or external
services. All code and interface text remain English. No remote publication is part of this
implementation request. The existing workspace plan is historical and is not overwritten.

## Data boundaries and architecture

1. Preserve the stratified 60% training / 15% validation / 25% final-test split.
2. Select the original model by validation ROC-AUC as before.
3. Clone that entire preprocessing/classification pipeline into CalibratedClassifierCV with
   sigmoid calibration and shuffled, seeded stratified folds on training rows only.
4. Use `ensemble=False`: out-of-fold training predictions fit the calibrator, and a final
   base pipeline is refitted on all training rows. Preprocessing is refitted within each fold.
5. Fix five folds, reduced only when the minority training class has fewer than five rows;
   require at least two rows per class. No method/fold search uses validation or test scores.
6. Report original/candidate quality on validation and final test separately. Do not display
   a data-dependent winner or tune anything using final-test metrics.
7. Keep the active pipeline unchanged. Persist the comparison candidate under a separate
   artifact key and bump artifact schema to v5 so old local artifacts regenerate safely.

`src/calibration.py` owns typed metric/bin calculations and training-only calibration.
`src/model.py` integrates aggregated reports and persistence. `app/calibration_view.py` renders
the comparison inside the existing evaluation screen. Tests belong in `tests/`.

Reliability bins use [lower, upper), except the final bin includes 1. Empty bins have count
zero and null means/rates; do not invent observed rates or connect those bins as observations.
Retain sample counts, and explain that small bins are noisy. Brier/log loss measure overall
probability quality, not calibration in isolation; neither establishes treatment effects.

## Acceptance criteria and implementation checklist

- [x] Metrics and bins: reject invalid/non-finite input, preserve boundary scores, include all
  bins, and serialize empty bins as strict JSON null. Verify `pytest tests/test_calibration.py`.
- [x] Training/report integration: candidate fits training-only folds, the original predictor
  and policy remain unchanged, validation/test reports reconcile, and both estimators survive
  a joblib round trip. Verify model and heldout-label regression tests.
- [x] Dashboard/reporting: show separately labeled split comparisons, reliability points,
  counts, diagnostic-only status, and downloadable aggregate results. Changing the diagnostic
  view must not alter single-customer or campaign predictions. Verify Streamlit AppTest.
- [x] Full pytest, Ruff, dependency consistency, real Telco smoke, and documentation review.
  Browser visual verification is recorded separately from Python-side UI tests.

## Local commands

```powershell
& .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --tb=short
& .\.venv\Scripts\python.exe -B -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Use small typed functions such as `probability_quality(labels, scores, n_bins=10)`.
Local changesets: metrics/training with tests; evaluation UI with tests; documentation
and recorded verification. Do not weaken or remove existing checks.

## Local verification: 2026-11-13

- Baseline: 82 tests passed before this change. Updated suite: 107 tests passed in 29.88 seconds,
  with no skips.
- Ruff and `pip check` passed. No dependencies or remote workflows were changed.
- The actual included Telco dataset was retrained with seed 42. The original selected model
  remains Logistic Regression, with unchanged raw classification metrics and policy behavior.
- The independent 1,761-row final-test comparison gave Brier 0.1665409250 -> 0.1365986328 and
  log loss 0.4921560421 -> 0.4158878996 for original -> sigmoid. This is one run, not a guarantee.
- Joblib round-trip tests verify both original and candidate estimators. A heldout-label
  inversion leaves validation comparisons and threshold recommendations unchanged.
- Streamlit AppTest checks split selection, unchanged live predictions, and report download.
- A real in-app browser rendered both split comparisons, table values, the fixed-domain
  reliability diagram, and sample-count descriptions. No JavaScript errors were observed.
  Existing threshold/global-chart Vega warnings appeared during rerenders; this is not a
  clean-console or full accessibility certification. Mobile breakpoints and Docker were not
  verified in this update. The local screenshot is excluded from version control.
- Publication and remote checks are not part of this local implementation.
- The loopback-only test server was stopped after verification; the screenshot remains in the
  ignored local `work/` directory. No customer records or new runtime dependencies were added.

## References

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
