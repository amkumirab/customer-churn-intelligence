"""Render the fixed-capacity evaluation for the active model."""

import pandas as pd
import streamlit as st


def render_ranking_evaluation(report: dict) -> None:
    st.subheader("Churn ranking at fixed outreach capacities")
    ranking = report.get("ranking")
    if ranking is None:
        st.info("Retrain with python -m src.train and restart the app to add ranking results.")
        return

    st.caption(
        "Customers are ranked by the active model score without a risk threshold. "
        "Capacity is rounded down to whole slots. At a tied cutoff, churners found "
        "is averaged across possible selections from the tie."
    )
    rows = pd.DataFrame([
        {"split": label, **row}
        for label, key in (("Validation", "validation"), ("Final test", "test"))
        for row in ranking[key]
    ])
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={
            "split": "Split",
            "capacity_pct": st.column_config.NumberColumn("Capacity (%)"),
            "rows": "Customers",
            "selected": "Slots",
            "total_churners": "Observed churners",
            "captured_churners": st.column_config.NumberColumn(
                "Churners found", format="%.2f"
            ),
            "churn_capture": st.column_config.NumberColumn("Churn capture", format="percent"),
            "random_capture": st.column_config.NumberColumn("Random capture", format="percent"),
            "lift": st.column_config.NumberColumn("Lift vs. random", format="%.2f"),
        },
    )
    st.caption(
        "Random capture is the expected share of observed churners found by selecting the "
        "same number of customers uniformly at random. Lift is churn capture divided by "
        "random capture. Undefined values are blank. These fixed capacities are for reporting "
        "only; the results do not measure retention impact or change the campaign policy."
    )
