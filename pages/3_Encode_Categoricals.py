# pages/3_🧮_Encode_Categoricals.py
import streamlit as st
import pandas as pd

from utils.state_manager import init_session_state, get_processed_df

def main():
    init_session_state()
    st.title("🧮 Step 3 – Encode Categorical Variables")

    df = get_processed_df()
    if df is None:
        st.warning("No data found. Please go to **Step 1 – Load Data** first.")
        return

    st.subheader("Current Columns & Data Types")
    dtypes_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str)
    })
    st.dataframe(dtypes_df)

    # Detect categorical columns
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "category"]

    if not cat_cols:
        st.info("No categorical (object/category) columns detected. Nothing to encode.")
        return

    selected_cols = st.multiselect(
        "Select categorical columns to encode using pd.get_dummies:",
        options=cat_cols,
        default=cat_cols  # you can change this if you want
    )

    drop_first = st.checkbox(
        "Drop first dummy for each selected column (helps with multicollinearity)",
        value=True
    )

    if st.button("Apply Encoding"):
        if not selected_cols:
            st.warning("Please select at least one column to encode.")
            return

        before_cols = list(df.columns)

        encoded_df = pd.get_dummies(
            df,
            columns=selected_cols,
            drop_first=drop_first,
            dtype=int   # ensures integer 0/1 for dummy columns
        )

        # Update only the processed_df in session_state
        st.session_state["processed_df"] = encoded_df

        st.success("✅ Encoding applied and stored in the current session.")

        st.write("Number of columns **before** encoding:", len(before_cols))
        st.write("Number of columns **after** encoding:", encoded_df.shape[1])

        with st.expander("Show new dummy columns"):
            new_cols = [c for c in encoded_df.columns if c not in before_cols]
            st.write(new_cols)

        st.markdown("➡️ Next: Go to **Step 4 – Train/Test Split** from the sidebar.")

if __name__ == "__main__":
    main()
