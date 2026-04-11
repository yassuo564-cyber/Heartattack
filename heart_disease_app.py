import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ================= CONSTANTS =================
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

# ================= LOAD =================
ann_model = load("ann_model.joblib")
knn_model = load("knn_model.joblib")
encoder = load("encoder.joblib")
scaler = load("scaler.joblib")

# ================= RADAR CHART =================
def plot_radar_chart():
    labels = np.array(["Accuracy", "Precision", "Recall", "F1", "AUC"])

    ann_vals = np.array([ANN_ACCURACY, ANN_PRECISION, ANN_RECALL, ANN_F1, ANN_AUC])
    knn_vals = np.array([KNN_ACCURACY, KNN_PRECISION, KNN_RECALL, KNN_F1, KNN_AUC])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    ann_vals = np.concatenate((ann_vals, [ann_vals[0]]))
    knn_vals = np.concatenate((knn_vals, [knn_vals[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, ann_vals, linewidth=2, label="ANN")
    ax.fill(angles, ann_vals, alpha=0.2)

    ax.plot(angles, knn_vals, linewidth=2, label="KNN")
    ax.fill(angles, knn_vals, alpha=0.2)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0.8, 1.0)

    ax.set_title("Model Performance Radar Chart")
    ax.legend(loc="upper right")

    return fig

# ================= STYLE =================
st.markdown("""
<style>
.stTabs [aria-selected="true"] {
    background-color: #2196F3;
    color: white;
}
.stButton > button {
    background-color: #2196F3;
    color: white;
}
h1 {color: #2196F3;}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("Heart Disease Prediction System")
st.caption("Supervised Machine Learning | ANN & KNN | Cleveland Dataset")
st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info"]
)

# ================= HOME =================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.16%")

# ================= PREDICTION =================
with tab2:
    st.subheader("Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.selectbox("Chest Pain", [0,1,2,3])
    trestbps = st.number_input("BP", 80, 200, 120)
    chol = st.number_input("Chol", 100, 600, 200)
    fbs = st.selectbox("FBS", [0,1])
    restecg = st.selectbox("ECG", [0,1,2])
    thalach = st.number_input("Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exang", [0,1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("CA", [0,1,2,3])
    thal = st.selectbox("Thal", [3,6,7])

    if st.button("Predict"):
        data = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

        ann_result = ann_model.predict(data)
        knn_result = knn_model.predict(pd.DataFrame(data))

        st.write("ANN:", ann_result[0])
        st.write("KNN:", knn_result[0])

# ================= MODEL =================
with tab3:
    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        df = pd.DataFrame({
            'Metric':['Accuracy','Precision','Recall','F1','AUC'],
            'ANN':[ANN_ACCURACY,ANN_PRECISION,ANN_RECALL,ANN_F1,ANN_AUC],
            'KNN':[KNN_ACCURACY,KNN_PRECISION,KNN_RECALL,KNN_F1,KNN_AUC]
        })
        st.dataframe(df)

    with col2:
        st.pyplot(plot_radar_chart())

    st.markdown("---")

    fig, ax = plt.subplots()
    sns.heatmap(ANN_CM, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# ================= MEMBER =================
with tab4:
    st.write("ANN - Ng Soon Siang")
    st.write("KNN - Chia Sheng Yang")

# ================= DATASET =================
with tab5:
    st.write("UCI Cleveland Dataset")
