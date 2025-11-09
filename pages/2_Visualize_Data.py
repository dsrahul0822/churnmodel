# pages/2_📊_Visualize_Data.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.state_manager import init_session_state, get_processed_df

def main():
    init_session_state()
    st.title("📊 Step 2 – Visualize Data")

    df = get_processed_df()
    if df is None:
        st.warning("No data found. Please go to **Step 1 – Load Data** first.")
        return

    st.write("Select a column to visualize:")

    col_name = st.selectbox("Column", df.columns)

    if not col_name:
        return

    col_data = df[col_name]

    # Detect type
    if col_data.dtype == "object" or str(col_data.dtype) == "category":
        # Categorical → Count plot
        st.subheader(f"Count Plot – {col_name}")
        value_counts = col_data.value_counts().sort_index()

        fig, ax = plt.subplots()
        value_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        # Numeric → Histogram
        st.subheader(f"Histogram – {col_name}")
        fig, ax = plt.subplots()
        ax.hist(col_data.dropna(), bins=30)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.info("For categorical columns you get a **count plot**, for numeric columns you get a **histogram**.")

if __name__ == "__main__":
    main()
