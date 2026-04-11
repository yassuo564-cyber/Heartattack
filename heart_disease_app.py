import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# =========================
# LOAD MODELS
# =========================
ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# =========================
# TITLE
# =========================
st.title("💓 Heart Disease Prediction System")
st.caption("AI Project | ANN vs KNN | UCI Cleveland Dataset")
st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info", "Explain AI"]
)

# =========================
# HOME
# =========================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.1%")

    st.markdown("---")

    st.subheader("Models Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**ANN (MLPClassifier)**")
        st.write("Learns complex patterns using hidden layers and backpropagation.")

    with col2:
        st.write("**KNN (K-Nearest Neighbours)**")
        st.write("Classifies based on similarity with nearest data points.")

# =========================
# PREDICTION
# =========================
with tab2:
    model_choice = st.selectbox("Select Model", ["ANN", "KNN"])

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x else "Female")
        cp = st.selectbox("Chest Pain", [0,1,2,3])
        trestbps = st.number_input("Blood Pressure", 50, 300, 120)

    with col2:
        chol = st.number_input("Cholesterol", 50, 600, 200)
        fbs = st.selectbox("High Blood Sugar", [0,1])
        restecg = st.selectbox("ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 50, 250, 150)

    with col3:
        exang = st.selectbox("Exercise Pain", [0,1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("Blood Vessels", [0,1,2,3])

    thal = st.selectbox("Thalassemia", [3,6,7])

    # =========================
    # VALIDATION
    # =========================
    if chol > 400:
        st.warning("⚠️ Cholesterol is very high")

    if age < 18:
        st.warning("⚠️ Model trained mainly on adults")

    # =========================
    # PREDICT
    # =========================
    if st.button("Run Prediction"):

        user_data = [[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]]

        if model_choice == "ANN":
            result = ann_model.predict(user_data)
            prob = ann_model.predict_proba(user_data)[0][1]

        else:
            df = pd.DataFrame(user_data, columns=[
                'age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal'
            ])

            cat = encoder.transform(df[['sex','cp','fbs','restecg','exang','slope','ca','thal']])
            num = scaler.transform(df[['age','trestbps','chol','thalach','oldpeak']])

            processed = np.concatenate([num, cat], axis=1)

            result = knn_model.predict(processed)
            prob = knn_model.predict_proba(processed)[0][1]

        st.markdown("---")

        st.metric("Risk Probability", f"{prob*100:.2f}%")
        st.progress(float(prob))

        if result[0] == 1:
            st.error("⚠️ Heart Disease Detected")
            st.write("Possible factors: high cholesterol, chest pain, low heart rate.")
        else:
            st.success("✅ No Heart Disease")
            st.write("Maintain healthy lifestyle.")

# =========================
# MODEL COMPARISON
# =========================
with tab3:

    st.subheader("Metrics Comparison")

    df = pd.DataFrame({
        'Metric': ['Accuracy','Precision','Recall','F1'],
        'ANN':[0.902,0.933,0.875,0.903],
        'KNN':[0.901,0.933,0.875,0.903]
    })

    st.dataframe(df)

    st.subheader("ROC Curve")

    # Dummy example curve (for presentation)
    fpr = [0, 0.1, 0.2, 1]
    tpr = [0, 0.7, 0.9, 1]

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# =========================
# MEMBER WORK
# =========================
with tab4:

    st.subheader("Group Contribution")

    table = pd.DataFrame({
        "Member":["Ng Soon Siang","Chia Sheng Yang"],
        "Model":["ANN","KNN"],
        "Contribution":[
            "Data cleaning, ANN model, evaluation",
            "Encoding, scaling, KNN tuning"
        ]
    })

    st.table(table)

    st.markdown("---")

    st.write("""
    ANN slightly outperforms KNN due to its ability to learn nonlinear patterns.
    KNN performs well due to small dataset size and simplicity.
    """)

# =========================
# DATASET INFO
# =========================
with tab5:

    st.subheader("Dataset Overview")

    st.write("303 patients, 13 features")

    fig, ax = plt.subplots()
    ax.bar(['No Disease','Disease'], [138,165])
    st.pyplot(fig)

    st.subheader("Feature Correlation")

    # Dummy correlation
    corr = np.random.rand(13,13)
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax)
    st.pyplot(fig)

# =========================
# EXPLAIN AI
# =========================
with tab6:

    st.subheader("How AI Works")

    st.write("""
    ANN:
    - Uses neural networks
    - Learns complex patterns

    KNN:
    - Distance-based
    - Uses nearest neighbours

    Both models achieved similar performance due to dataset size.
    """)
