# utils/state_manager.py
import streamlit as st
import pandas as pd

def init_session_state():
    """
    Initialize all the keys in st.session_state that we will use
    across different Streamlit pages.
    """
    default_keys = {
        "raw_df": None,          # Original loaded data
        "processed_df": None,    # Working copy after transformations
        "target_column": None,   # Selected target for prediction (e.g., 'Exited')
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "model": None,           # Trained LogisticRegression model
        "feature_columns": None  # List of feature column names used for training
    }

    for key, value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_dataframes(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    """
    Helper to update raw_df and processed_df in session_state.
    """
    st.session_state["raw_df"] = raw_df
    st.session_state["processed_df"] = processed_df


def get_raw_df():
    return st.session_state.get("raw_df", None)


def get_processed_df():
    return st.session_state.get("processed_df", None)
