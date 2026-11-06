# Implementation plan: Retention Campaign Workspace

The user delegated local implementation and initially deferred Git/GitHub operations.
Publication as eight separate commits was subsequently authorized on 2026-11-06: preserve
the four existing repair commits and add four focused workspace commits without squashing.
Scope is defined in `SPEC.md`; tasks are tracked in `tasks/todo.md`.

1. Establish deterministic campaign allocation with failing unit tests, then implement it.
2. Add identity-safe cohort scoring and segment summaries, verified against reordered inputs.
3. Add an in-memory decision bundle, preserving the existing spreadsheet export protection.
4. Integrate focused dashboard views; preserve single-customer and sidebar behavior.
5. Review edge cases, run local tests/lint/runtime checks, and prepare tomorrow's checklist.

The interface uses the existing Streamlit design system: labeled native controls, restrained
tables and charts, three task-oriented tabs, no injected HTML/CSS, no custom palette. Charts
have adjacent numerical tables. Empty and zero-budget states are actionable rather than blank.

Risks: class-balanced scores are not calibrated probabilities; scenario success rates are
assumptions, not measured treatment effects. Revenue is not profit or lifetime value. Uploaded
IDs must be unique to prevent accidental repeated outreach. No persistence or network side
effects are introduced. Constant contact cost makes a sorted top-N allocation sufficient;
there is no need for an optimization framework or database.
