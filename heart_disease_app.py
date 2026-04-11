import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

# ===== Page Config =====
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ===== Load Models (Cloud-safe) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ann_model = load(os.path.join(BASE_DIR, 'ann_model.joblib'))
knn_model = load(os.path.join(BASE_DIR, 'knn_model.joblib'))
encoder = load(os.path.join(BASE_DIR, 'encoder.joblib'))
scaler = load(os.path.join(BASE_DIR, 'scaler.joblib'))

# ===== Load Dataset =====
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=cols, na_values='?')
    df = df.fillna(df.median())
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()

# ===== CSS (你原本的，不动) =====
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 5px; }
    .stTabs [aria-selected="true"] { background-color: #2196F3; color: white; }
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton > button:hover { background-color: #1976D2; }
    div[data-testid="stMetric"] {
        background-color: #1E1E2E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .block-container { padding-top: 2rem; }
    h1 { color: #2196F3; }
</style>
""", unsafe_allow_html=True)

# ===== Title =====
st.title("Heart Disease Prediction System")
st.caption("Supervised Machine Learning | ANN & KNN | Cleveland Dataset")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info"]
)

# =====================
# HOME
# =====================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.1%")

# =====================
# PREDICTION（原封不动）
# =====================
with tab2:
    model_choice = st.selectbox("Model", ["ANN", "KNN"])

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Gender", [1, 0])
    cp = st.selectbox("Chest Pain", [0,1,2,3])
    trestbps = st.number_input("Blood Pressure", 50, 300, 120)
    chol = st.number_input("Cholesterol", 50, 600, 200)
    fbs = st.selectbox("High Sugar", [0,1])
    restecg = st.selectbox("ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    exang = st.selectbox("Exercise Pain", [0,1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Vessels", [0,1,2,3])
    thal = st.selectbox("Thal", [3,6,7])

    if st.button("Run Prediction"):
        user = [[age, sex, cp, trestbps, chol, fbs, restecg,
                 thalach, exang, oldpeak, slope, ca, thal]]

        if model_choice == "ANN":
            result = ann_model.predict(user)
        else:
            df_user = pd.DataFrame(user, columns=df.columns[:-1])
            cat = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
            num = ['age','trestbps','chol','thalach','oldpeak']
            df_cat = pd.DataFrame(encoder.transform(df_user[cat]))
            df_num = pd.DataFrame(scaler.transform(df_user[num]))
            final = pd.concat([df_num, df_cat], axis=1)
            result = knn_model.predict(final)

        if result[0] == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease")

# =====================
# MODEL COMPARISON（增强🔥）
# =====================
with tab3:
    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        'Metric': ['Accuracy','Precision','Recall','F1'],
        'ANN':[0.902,0.933,0.875,0.903],
        'KNN':[0.9016,0.9333,0.875,0.9032]
    })

    st.dataframe(comparison)

    fig, ax = plt.subplots()
    comparison.set_index("Metric").plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Correlation
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    st.pyplot(fig)

    # Distribution
    st.subheader("Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax)
    st.pyplot(fig)

    # ROC
    st.subheader("ROC Curve")
    X = df.drop('target', axis=1)
    y = df['target']
    prob = ann_model.predict_proba(X)[:,1]

    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    st.pyplot(fig)

# =====================
# MEMBER WORK（原本）
# =====================
with tab4:
    st.write("Your original content here")

# =====================
# DATASET INFO（增强🔥）
# =====================
with tab5:
    st.subheader("Feature Importance")

    importance = df.corr()['target'].abs().sort_values(ascending=False)[1:]

    fig, ax = plt.subplots()
    importance.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Pairplot")

    sample = df[['age','chol','thalach','oldpeak','target']].sample(100)
    fig = sns.pairplot(sample, hue='target')
    st.pyplot(fig)
