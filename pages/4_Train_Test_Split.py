# pages/4_✂️_Train_Test_Split.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.state_manager import init_session_state, get_processed_df

def main():
    init_session_state()
    st.title("✂️ Step 4 – Train/Test Split")

    df = get_processed_df()
    if df is None:
        st.warning("No data found. Please go to **Step 1 – Load Data** first.")
        return

    st.write("Shape of current processed data:", df.shape)

    # Try to default to 'Exited' if present
    if "Exited" in df.columns:
        default_index = df.columns.get_loc("Exited")
    else:
        default_index = len(df.columns) - 1

    target_col = st.selectbox(
        "Select target column (what you want to predict – e.g., churn flag):",
        options=df.columns,
        index=default_index
    )

    train_size_percent = st.slider(
        "Training data percentage",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        help="Example: 70% train / 30% test"
    )

    if st.button("Split into Train and Test"):
        if not target_col:
            st.warning("Please select a target column.")
            return

        # 1️⃣ Separate X and y
        X_full = df.drop(columns=[target_col])
        y = df[target_col]

        # 2️⃣ Ensure target is numeric
        if y.dtype == "object":
            y = y.astype("category").cat.codes
            st.info("Target column was non-numeric; converted to category codes for modelling.")

        # 3️⃣ Keep ONLY numeric/bool columns for X
        numeric_cols = X_full.select_dtypes(include=["number", "bool"]).columns.tolist()
        non_numeric_cols = [c for c in X_full.columns if c not in numeric_cols]

        if non_numeric_cols:
            st.warning(
                "The following non-numeric columns will be **dropped** from the model input, "
                "because Logistic Regression requires numeric features.\n\n"
                f"👉 {non_numeric_cols}\n\n"
                "If you want to use them, please encode them in **Step 3 – Encode Categorical Variables**."
            )

        X = X_full[numeric_cols]

        train_size = train_size_percent / 100.0

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )

        # Save in session_state
        st.session_state["target_column"] = target_col
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.success("✅ Train/Test split completed and saved in session.")
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"X_test shape: {X_test.shape}")

        with st.expander("y_train distribution (normalized)"):
            st.write(y_train.value_counts(normalize=True).round(3))

        with st.expander("Final feature columns used for the model"):
            st.write(numeric_cols)

        st.markdown("➡️ Next: Go to **Step 5 – Train & Evaluate Model**.")

if __name__ == "__main__":
    main()
