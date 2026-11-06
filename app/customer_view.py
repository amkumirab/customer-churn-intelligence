"""Single-customer profile review and local explanations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explain import explain_customer
from src.model import predict_customer


def render_customer_review(
    artifact: dict, high_risk_threshold: float, medium_risk_threshold: float
) -> None:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Customer profile")
        with st.form("customer_form"):
            c1, c2, c3 = st.columns(3)
            tenure = c1.slider("Tenure (months)", 0, 72, 12)
            monthly = c2.number_input("Monthly charges ($)", 18.0, 125.0, 79.5, step=1.0)
            total = c3.number_input(
                "Total charges ($)", 0.0, 10_000.0, float(round(monthly * tenure, 2)), step=25.0
            )
            contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = c2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            payment = c3.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Credit card (automatic)",
                    "Bank transfer (automatic)",
                    "Mailed check",
                ],
            )
            paperless = c1.selectbox("Paperless billing", ["Yes", "No"])
            senior = c2.selectbox("Senior citizen", ["No", "Yes"])
            gender = c3.selectbox("Gender", ["Female", "Male"])
            partner = c1.selectbox("Has a partner", ["No", "Yes"])
            dependents = c2.selectbox("Has dependents", ["No", "Yes"])
            phone = c3.selectbox("Phone service", ["Yes", "No"])

            with st.expander("Service subscriptions"):
                s1, s2, s3 = st.columns(3)
                multiple_lines = s1.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
                online_security = s2.selectbox(
                    "Online security", ["No", "Yes", "No internet service"]
                )
                online_backup = s3.selectbox("Online backup", ["No", "Yes", "No internet service"])
                device_protection = s1.selectbox(
                    "Device protection", ["No", "Yes", "No internet service"]
                )
                tech_support = s2.selectbox("Tech support", ["No", "Yes", "No internet service"])
                streaming_tv = s3.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = s1.selectbox(
                    "Streaming movies", ["No", "Yes", "No internet service"]
                )
            submitted = st.form_submit_button("Calculate churn risk", width="stretch")

    with right:
        st.subheader("Risk assessment")
        if submitted:
            st.session_state["customer_profile"] = {
                "tenure_months": tenure,
                "monthly_charges": monthly,
                "total_charges": total,
                "gender": gender,
                "senior_citizen": senior,
                "has_partner": partner,
                "has_dependents": dependents,
                "phone_service": phone,
                "multiple_lines": multiple_lines,
                "internet_service": internet,
                "online_security": online_security,
                "online_backup": online_backup,
                "device_protection": device_protection,
                "tech_support": tech_support,
                "streaming_tv": streaming_tv,
                "streaming_movies": streaming_movies,
                "contract_type": contract,
                "paperless_billing": paperless,
                "payment_method": payment,
            }
        if "customer_profile" in st.session_state:
            customer = st.session_state["customer_profile"]
            result = predict_customer(
                artifact["pipeline"],
                customer,
                high_risk_threshold=high_risk_threshold,
                medium_risk_threshold=medium_risk_threshold,
            )
            probability = result["churn_probability"]
            st.metric("Predicted churn probability", f"{probability:.1%}")
            st.progress(probability)
            if result["risk_level"] == "High":
                st.error(f"High risk — {result['action']}")
            elif result["risk_level"] == "Medium":
                st.warning(f"Medium risk — {result['action']}")
            else:
                st.success(f"Low risk — {result['action']}")

            local_explanation = explain_customer(
                artifact["pipeline"],
                customer,
                artifact["explanation_background"],
            )
            st.markdown("**Why this prediction?**")
            st.caption(
                "Positive SHAP values increase predicted churn risk; negative values reduce it."
            )
            st.bar_chart(local_explanation.set_index("feature")[["impact"]])
            st.dataframe(
                local_explanation,
                hide_index=True,
                width="stretch",
                column_config={
                    "feature": "Factor",
                    "value": "Customer value",
                    "impact": st.column_config.NumberColumn("SHAP impact", format="%.4f"),
                    "direction": "Effect",
                },
            )

            st.markdown("**Submitted profile**")
            st.dataframe(pd.DataFrame([customer]), hide_index=True, width="stretch")
        else:
            st.info("Complete the profile and calculate its risk to see a prediction.")
