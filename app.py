# app.py
import streamlit as st
from utils.state_manager import init_session_state

st.set_page_config(
    page_title="Customer Churn – Logistic Regression Demo",
    layout="wide"
)

def main():
    # Initialize all session_state keys we’ll use across pages
    init_session_state()

    st.title("📉 Customer Churn Prediction – Logistic Regression (Demo App)")

    st.markdown("""
    Welcome to the **Customer Churn Prediction** demo built with **Streamlit** and **Logistic Regression**.

    ### 🔁 Workflow Steps

    1. **Load Data** (Page 1)  
       - Load the churn dataset (from file or default CSV).  
       - Preview rows & basic info.

    2. **Visualize Data** (Page 2)  
       - Create **count plots** for categorical features.  
       - Create **histograms** for numerical features.

    3. **Encode Categorical Variables** (Page 3)  
       - Use `pd.get_dummies` on selected columns.  
       - Option to **drop first** dummy to handle multicollinearity.  
       - Ensure encoded columns are numeric.

    4. **Train/Test Split** (Page 4)  
       - Select target column (e.g., `Exited`).  
       - Choose train/test split percentages.

    5. **Train & Evaluate Model** (Page 5)  
       - Train **Logistic Regression**.  
       - View **accuracy**, **confusion matrix**, **precision**, **recall**, and **F1-score**.

    6. **Predict New Customer** (Page 6)  
       - Manually enter feature values for a single customer.  
       - Get **probability of churn** and a clear message:
         - ✅ *“Customer will NOT churn with probability XX%”*  
         - ⚠️ *“Customer WILL churn with probability YY%”*

    ---

    Use the **sidebar** to navigate between pages.
    """)

if __name__ == "__main__":
    main()
