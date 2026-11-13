# Changelog

## Unreleased

### Added

- Training-only sigmoid calibration comparison with Brier score, log loss, reliability points,
  sample counts, and a downloadable aggregate JSON report. The active model is unchanged.
- Retention campaign workspace with risk/contract filters and operational cohort segments.
- Budget- and capacity-aware contact selection with risk or scenario-value prioritization.
- Editable retention assumptions, positive-net-value eligibility, and budget sensitivity.
- In-memory ZIP decision bundles with protected CSVs and a provenance manifest.
- Model evaluation view with validation/test comparisons, policy confusion counts, and charts.
- Sample-data workflow and unit, integration, and dashboard regression tests.

### Changed

- Artifact schema v5 stores the comparison candidate separately and regenerates older local
  artifacts on dashboard launch.
- Dashboard responsibilities are separated into focused customer, campaign, and evaluation views.
- Batch scores retain full precision to keep eligibility consistent with risk classification.
- Missing identifiers are rejected before nullable values can become literal identifier strings.
- Protected CSV exports use fixed LF line endings for platform-independent cohort fingerprints.

Publication is subject to repository checks; manual visual and container validation remain
separate from the automated test suite.
