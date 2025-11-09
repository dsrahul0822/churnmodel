# pages/1_📥_Load_Data.py
import os
import streamlit as st
import pandas as pd

from utils.state_manager import init_session_state, set_dataframes

def load_default_data():
    """
    Load the default churn dataset from the data folder.
    Adjust the path if your file name/path is different.
    """
    data_path = os.path.join("data", "Churn_Modelling.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error(f"Default dataset not found at: {data_path}")
        return None

def main():
    init_session_state()

    st.title("📥 Step 1 – Load Churn Dataset")

    st.markdown("""
    You can either:
    - **Upload a CSV file**, or  
    - Use the **default demo file**: `data/Churn_Modelling.csv`
    """)

    # Option 1: Upload CSV
    uploaded_file = st.file_uploader("Upload your churn CSV file", type=["csv"])

    col1, col2 = st.columns(2)

    with col1:
        use_default = st.button("Use Default Demo Dataset")

    df = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    elif use_default:
        df = load_default_data()
        if df is not None:
            st.success("✅ Loaded default dataset from data/Churn_Modelling.csv")

    # If we already have data in session_state, show info and allow reuse
    if df is None and st.session_state["raw_df"] is not None:
        st.info("Using dataset already loaded in this session.")
        df = st.session_state["raw_df"]

    if df is not None:
        # Update session_state
        set_dataframes(raw_df=df, processed_df=df.copy())

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        st.write("**Shape:** ", df.shape)

        with st.expander("Column Info"):
            dtypes_df = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str)
            })
            st.dataframe(dtypes_df)

        st.success("✅ Data stored in session and ready for the next steps.")
        st.markdown("➡️ Now go to **Page 2 – Visualize Data** from the sidebar.")
    else:
        st.warning("Please upload a CSV file or click **Use Default Demo Dataset** to proceed.")

if __name__ == "__main__":
    main()
