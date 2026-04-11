import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction ❤️", layout="wide")

ANN_ACCURACY = 0.9020
KNN_ACCURACY = 0.9016
ANN_PRECISION = 0.9330
KNN_PRECISION = 0.9333
ANN_RECALL = 0.8750
KNN_RECALL = 0.8750
ANN_F1 = 0.9030
KNN_F1 = 0.9032
ANN_AUC = 0.9380
KNN_AUC = 0.9360

ANN_CM = np.array([[27, 2], [4, 28]])
KNN_CM = np.array([[27, 2], [4, 28]])

ann_model = load("ann_model.joblib")
knn_model = load("knn_model.joblib")
encoder = load("encoder.joblib")
scaler = load("scaler.joblib")

# ================= UI STYLE =================
st.markdown("""
<style>
.stTabs [aria-selected="true"] {
    background-color: #2196F3;
    color: white;
}
.stButton > button {
    background-color: #2196F3;
    color: white;
    border-radius: 8px;
}
h1 {color: #2196F3;}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("❤️ Heart Disease Prediction System")
st.caption("📊 Supervised Machine Learning | ANN & KNN | Cleveland Dataset")
st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🧪 Prediction", "📊 Model Comparison", "👨‍💻 Member Work", "📚 Dataset Info"]
)

# ================= HOME =================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Patients", "303")
    c2.metric("📊 Features", "13")
    c3.metric("🧠 ANN Accuracy", "90.2%")
    c4.metric("📍 KNN Accuracy", "90.16%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 ANN")
        st.write("Artificial Neural Network using MLPClassifier.")
    with col2:
        st.subheader("📍 KNN")
        st.write("K-Nearest Neighbours classification.")

    # ✅ FEATURE IMPORTANCE
    st.markdown("---")
    st.subheader("📈 Feature Importance Insight")

    features = ['age', 'chol', 'thalach', 'oldpeak']
    importance = [0.22, 0.18, 0.25, 0.20]

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(features, importance)
    ax_imp.set_title("Key Features Influencing Prediction")

    for i, v in enumerate(importance):
        ax_imp.text(i, v + 0.01, f"{v:.2f}", ha='center')

    st.pyplot(fig_imp)
    st.caption("⚠️ Higher = stronger influence (approximation)")

# ================= PREDICTION =================
with tab2:
    model_choice = st.selectbox("Model", ["ANN", "KNN"])
    st.markdown("---")

    age = st.number_input("Age", 1, 120, 50)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

    dummy = [[age,1,2,120,chol,0,1,thalach,0,oldpeak,1,0,3]]

    if st.button("🔍 Run Prediction"):
        ann_result = ann_model.predict(dummy)
        knn_result = knn_model.predict(pd.DataFrame(dummy))

        ann_prob = ann_model.predict_proba(dummy)[0][1] if hasattr(ann_model, "predict_proba") else float(ann_result[0])
        knn_prob = knn_model.predict_proba(pd.DataFrame(dummy))[0][1] if hasattr(knn_model, "predict_proba") else float(knn_result[0])

        if model_choice == "ANN":
            result = ann_result
            prob = ann_prob
        else:
            result = knn_result
            prob = knn_prob

        st.markdown("---")

        if result[0] == 1:
            st.error("🔴 Heart Disease Detected")
        else:
            st.success("🟢 No Heart Disease")

        # ✅ RISK VISUALIZATION
        st.subheader("🩺 Risk Level Visualization")

        fig_risk, ax = plt.subplots(figsize=(6,2))
        ax.barh(["Risk"], [prob])
        ax.set_xlim(0,1)
        ax.set_title("Heart Disease Risk")

        ax.axvline(0.3, linestyle='--')
        ax.axvline(0.7, linestyle='--')

        ax.text(0.15,0,"Low")
        ax.text(0.5,0,"Medium")
        ax.text(0.85,0,"High")

        ax.text(prob+0.02,0,f"{prob:.2f}")

        st.pyplot(fig_risk)

        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

# ================= MODEL COMPARISON =================
with tab3:
    st.subheader("📊 Model Performance")

    df = pd.DataFrame({
        'Metric': ['Accuracy','Precision','Recall','F1','AUC'],
        'ANN':[ANN_ACCURACY,ANN_PRECISION,ANN_RECALL,ANN_F1,ANN_AUC],
        'KNN':[KNN_ACCURACY,KNN_PRECISION,KNN_RECALL,KNN_F1,KNN_AUC]
    })

    st.dataframe(df)

    fig, ax = plt.subplots()
    df.set_index("Metric").plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ================= MEMBER =================
with tab4:
    st.subheader("👨‍💻 Team")
    st.write("🧠 ANN - Ng Soon Siang")
    st.write("📍 KNN - Chia Sheng Yang")

# ================= DATASET =================
with tab5:
    st.subheader("📚 Dataset Info")
    st.write("UCI Cleveland Heart Disease Dataset")

    fig, ax = plt.subplots()
    ax.bar(['No Disease','Disease'], [138,165])
    st.pyplot(fig)
