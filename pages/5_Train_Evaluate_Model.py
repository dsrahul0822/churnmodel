# pages/5_🧠_Train_Evaluate_Model.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib

from utils.state_manager import init_session_state

def main():
    init_session_state()
    st.title("🧠 Step 5 – Train & Evaluate Logistic Regression Model")

    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    if X_train is None or X_test is None or y_train is None or y_test is None:
        st.warning("Train/Test data not found. Please complete **Step 4 – Train/Test Split** first.")
        return

    st.write("Training features shape:", X_train.shape)
    st.write("Test features shape:", X_test.shape)

    with st.expander("Feature columns used for training"):
        st.write(list(X_train.columns))

    C_value = st.slider(
        "Regularization strength (C) for Logistic Regression",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01
    )

    if st.button("Train Logistic Regression Model"):
        model = LogisticRegression(
            C=C_value,
            max_iter=1000,
            solver="lbfgs"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("📈 Evaluation Metrics")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")

        st.subheader("📊 Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0 (No Churn)", "Actual 1 (Churn)"],
            columns=["Predicted 0", "Predicted 1"]
        )
        st.dataframe(cm_df)

        # Confusion matrix heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix Heatmap")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)

        # Save model and feature columns in session
        st.session_state["model"] = model
        st.session_state["feature_columns"] = list(X_train.columns)

        # Save to disk as well (for persistence)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, os.path.join("models", "logistic_model.pkl"))
        joblib.dump(list(X_train.columns), os.path.join("models", "feature_columns.pkl"))

        st.success("✅ Model trained and saved (session + `models/` folder).")
        st.markdown("➡️ Next: Go to **Step 6 – Predict New Customer**.")

if __name__ == "__main__":
    main()
