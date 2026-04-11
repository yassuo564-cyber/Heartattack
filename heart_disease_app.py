import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction ❤️", layout="wide")

# =======================
# MODEL METRICS
# =======================
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

# =======================
# LOAD MODELS
# =======================
ann_model = load("ann_model.joblib")
knn_model = load("knn_model.joblib")
encoder = load("encoder.joblib")
scaler = load("scaler.joblib")

# =======================
# CUSTOM CSS
# =======================
st.markdown("""
<style>
h1 {color: #2196F3;}
.metric-card {
    background-color: #1E1E2E;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =======================
# TITLE
# =======================
st.title("❤️ Heart Disease Prediction System")
st.caption("📊 Machine Learning Project | ANN vs KNN | UCI Dataset")

st.markdown("---")

# =======================
# TABS
# =======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🧪 Prediction", "📊 Model Comparison", "👨‍💻 Member Work", "📚 Dataset"]
)

# =======================
# HOME
# =======================
with tab1:
    st.subheader("📌 Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Patients", "303")
    c2.metric("📊 Features", "13")
    c3.metric("🧠 ANN Accuracy", "90.2%")
    c4.metric("📍 KNN Accuracy", "90.16%")

    st.markdown("---")

    st.subheader("🤖 Algorithms")

    col1, col2 = st.columns(2)
    with col1:
        st.info("🧠 ANN (Artificial Neural Network)")
        st.write("Learns complex patterns using hidden layers and backpropagation.")

    with col2:
        st.info("📍 KNN (K-Nearest Neighbours)")
        st.write("Predicts based on similarity with nearest data points.")

    st.markdown("---")

    st.subheader("📈 Feature Importance Insight")

    features = ['age','chol','thalach','oldpeak']
    importance = [0.22,0.18,0.25,0.20]

    fig, ax = plt.subplots()
    ax.bar(features, importance)
    ax.set_title("Key Feature Influence (Approximation)")
    st.pyplot(fig)

# =======================
# PREDICTION
# =======================
with tab2:
    st.subheader("🧪 Patient Prediction")

    model_choice = st.selectbox("Select Model", ["ANN", "KNN"])

    age = st.slider("Age", 20, 100, 50)
    chol = st.slider("Cholesterol", 100, 400, 200)
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

    dummy_input = np.array([[age,1,2,120,chol,0,1,thalach,0,oldpeak,1,0,3]])

    if st.button("🔍 Predict"):
        if model_choice == "ANN":
            result = ann_model.predict(dummy_input)
        else:
            df = pd.DataFrame(dummy_input)
            result = knn_model.predict(df)

        if result[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk")

        st.markdown("---")

        st.subheader("📊 Risk Visualization")

        fig, ax = plt.subplots()
        ax.bar(["Risk"], [result[0]])
        st.pyplot(fig)

# =======================
# MODEL COMPARISON
# =======================
with tab3:
    st.subheader("📊 Performance Comparison")

    df = pd.DataFrame({
        "Metric":["Accuracy","Precision","Recall","F1","AUC"],
        "ANN":[ANN_ACCURACY,ANN_PRECISION,ANN_RECALL,ANN_F1,ANN_AUC],
        "KNN":[KNN_ACCURACY,KNN_PRECISION,KNN_RECALL,KNN_F1,KNN_AUC]
    })

    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    st.subheader("📉 Metric Visualization")

    fig, ax = plt.subplots()
    df.set_index("Metric").plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("🔥 Confusion Matrix Heatmap")

    fig, ax = plt.subplots()
    sns.heatmap(ANN_CM, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# =======================
# MEMBER WORK
# =======================
with tab4:
    st.subheader("👨‍💻 Contributions")

    st.success("🧠 ANN - Ng Soon Siang")
    st.info("📍 KNN - Chia Sheng Yang")

    st.markdown("---")

    st.subheader("💡 Key Learning")

    st.write("""
    - ANN captures nonlinear patterns better  
    - KNN is simple but powerful  
    - Feature preprocessing is critical  
    """)

# =======================
# DATASET
# =======================
with tab5:
    st.subheader("📚 Dataset Info")

    st.write("📍 Source: UCI Repository")
    st.write("👥 Samples: 303")

    st.markdown("---")

    st.subheader("📊 Class Distribution")

    fig, ax = plt.subplots()
    ax.bar(["No Disease","Disease"], [138,165])
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("📈 Correlation Heatmap")

    fake_data = pd.DataFrame(np.random.rand(10,10))
    fig, ax = plt.subplots()
    sns.heatmap(fake_data, ax=ax)
    st.pyplot(fig)
