# pages/6_Predict_New_Customer.py  (or 6_🔮_Predict_New_Customer.py)
import streamlit as st
import pandas as pd
import joblib

from utils.state_manager import init_session_state

def main():
    init_session_state()
    st.title("🔮 Step 6 – Predict New Customer Churn")

    model = st.session_state.get("model")
    feature_columns = st.session_state.get("feature_columns")

    # If model not in session, try loading from disk
    if model is None or feature_columns is None:
        try:
            model = joblib.load("models/logistic_model.pkl")
            feature_columns = joblib.load("models/feature_columns.pkl")
            st.session_state["model"] = model
            st.session_state["feature_columns"] = feature_columns
            st.info("Loaded model and feature columns from `models/` folder.")
        except Exception:
            st.warning(
                "No trained model found. Please train the model in **Step 5** "
                "before doing predictions."
            )
            return

    st.markdown("""
    Enter values for **one new customer** below.

    ⚠️ These inputs correspond to the **processed feature columns** used during training  
    (including dummy variables created in Step 3).
    """)

    inputs = {}

    with st.form("prediction_form"):
        for col in feature_columns:
            value = st.number_input(
                f"{col}",
                value=0.0,
                step=1.0
            )
            inputs[col] = value

        submitted = st.form_submit_button("Predict Churn Probability")

    if submitted:
        # Create a single-row DataFrame with the correct columns
        input_df = pd.DataFrame([inputs], columns=feature_columns)

        # ---------- 1️⃣ Probability Prediction ----------
        proba = model.predict_proba(input_df)[0]
        prob_not_churn = proba[0]
        prob_churn = proba[1]

        st.subheader("📊 Prediction Result")
        st.write(f"Probability of **NOT churn (0)**: `{prob_not_churn*100:.2f}%`")
        st.write(f"Probability of **churn (1)**: `{prob_churn*100:.2f}%`")

        if prob_churn >= 0.5:
            st.error(
                f"⚠️ The model predicts: **Customer WILL CHURN** "
                f"with probability `{prob_churn*100:.2f}%`."
            )
        else:
            st.success(
                f"✅ The model predicts: **Customer will NOT churn** "
                f"with probability `{prob_not_churn*100:.2f}%`."
            )

        # ---------- 2️⃣ Explanation: Why this prediction? ----------
        st.subheader("🔍 Why this prediction? (Top contributing features)")

        # For binary logistic regression: model.coef_[0] gives coefficients
        coef = model.coef_[0]   # shape: (n_features,)
        values = input_df.iloc[0].values

        # Contribution ≈ coefficient × value (in log-odds space)
        contributions = values * coef

        contrib_df = pd.DataFrame({
            "feature": feature_columns,
            "input_value": values,
            "coefficient": coef,
            "contribution": contributions
        })

        # Sort by contribution (descending: pushes towards churn)
        contrib_sorted = contrib_df.sort_values(by="contribution", ascending=False)

        top_n = 5  # how many features to show

        st.markdown("**Top features increasing churn risk (pushing towards 1):**")
        top_positive = contrib_sorted.head(top_n)
        if not top_positive.empty:
            st.dataframe(
                top_positive[["feature", "input_value", "coefficient", "contribution"]]
                .style.format({"input_value": "{:.2f}", "coefficient": "{:.4f}", "contribution": "{:.4f}"})
            )
        else:
            st.write("No strong positive contributors found.")

        st.markdown("**Top features decreasing churn risk (pushing towards 0):**")
        top_negative = contrib_sorted.sort_values(by="contribution", ascending=True).head(top_n)
        if not top_negative.empty:
            st.dataframe(
                top_negative[["feature", "input_value", "coefficient", "contribution"]]
                .style.format({"input_value": "{:.2f}", "coefficient": "{:.4f}", "contribution": "{:.4f}"})
            )
        else:
            st.write("No strong negative contributors found.")

        st.info(
            "💡 Interpretation:\n"
            "- **Coefficient**: how strongly this feature influences churn in general (positive → more churn, negative → less churn).\n"
            "- **Contribution**: coefficient × input_value for this customer (larger magnitude = more impact on this specific prediction).\n"
            "\nThis is a simple, log-odds based explanation ideal for demos. "
            "For production, you might consider SHAP or other explainability tools."
        )

if __name__ == "__main__":
    main()
