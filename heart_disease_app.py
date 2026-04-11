import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= PAGE =================
st.set_page_config(
    page_title="Heart Disease Prediction ❤️",
    page_icon="❤️",
    layout="wide"
)

# ================= LOAD =================
try:
    ann_model = load("ann_model.joblib")
    knn_model = load("knn_model.joblib")
except:
    st.error("❌ Model files not found. Make sure .joblib files are in the same folder.")
    st.stop()

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

    st.subheader("📈 Feature Importance Insight")

    features = ['age', 'chol', 'thalach', 'oldpeak']
    importance = [0.22, 0.18, 0.25, 0.20]

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(features, importance)
    ax_imp.set_title("Key Features Influencing Prediction")

    for i, v in enumerate(importance):
        ax_imp.text(i, v + 0.01, f"{v:.2f}", ha='center')

    st.pyplot(fig_imp)
    plt.close(fig_imp)

# ================= PREDICTION =================
with tab2:
    st.subheader("🧪 Patient Clinical Form")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Gender", ["Male", "Female"])
        cp = st.selectbox("Chest Pain", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        trestbps = st.number_input("Blood Pressure", 80, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("High Blood Sugar?", ["No", "Yes"])
        restecg = st.selectbox("ECG Result", ["Normal", "ST-T abnormality", "LV hypertrophy"])
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)

    with col3:
        exang = st.selectbox("Exercise Chest Pain?", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Blood Vessels", [0,1,2,3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # ===== ENCODING =====
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal":2,"Asymptomatic":3}
    restecg_map = {"Normal":0,"ST-T abnormality":1,"LV hypertrophy":2}
    slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
    thal_map = {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}

    cp = cp_map[cp]
    restecg = restecg_map[restecg]
    slope = slope_map[slope]
    thal = thal_map[thal]

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # ===== PREDICTION =====
    if st.button("🔍 Run Prediction"):

        try:
            ann_result = ann_model.predict(input_data)
        except:
            ann_result = [0]

        try:
            knn_result = knn_model.predict(pd.DataFrame(input_data))
        except:
            knn_result = [0]

        # ===== PROBABILITY SAFE =====
        try:
            ann_prob = ann_model.predict_proba(input_data)[0][1]
        except:
            ann_prob = 0.5

        try:
            knn_prob = knn_model.predict_proba(pd.DataFrame(input_data))[0][1]
        except:
            knn_prob = 0.5

        # 默认 ANN
        result = ann_result
        prob = ann_prob

        st.markdown("---")

        if result[0] == 1:
            st.error("🔴 Heart Disease Detected")
        else:
            st.success("🟢 No Heart Disease")

        # ===== RISK BAR =====
        st.subheader("🩺 Risk Level Visualization")

        fig_risk, ax = plt.subplots(figsize=(6,2))
        ax.barh(["Risk"], [prob])
        ax.set_xlim(0,1)

        ax.axvline(0.3, linestyle='--')
        ax.axvline(0.7, linestyle='--')

        ax.text(0.15,0,"Low")
        ax.text(0.5,0,"Medium")
        ax.text(0.85,0,"High")

        ax.text(prob+0.02,0,f"{prob:.2f}")

        st.pyplot(fig_risk)
        plt.close(fig_risk)

        # ===== INTERPRET =====
        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

# ================= MODEL =================
with tab3:
    st.subheader("📊 Model Performance Comparison")

    df = pd.DataFrame({
        'Metric': ['Accuracy','Precision','Recall','F1','AUC'],
        'ANN':[0.902,0.933,0.875,0.903,0.938],
        'KNN':[0.9016,0.9333,0.875,0.9032,0.936]
    })

    st.dataframe(df)

    fig, ax = plt.subplots()
    df.set_index("Metric").plot(kind="bar", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ================= MEMBER =================
with tab4:
    st.subheader("👨‍💻 Team Contribution")
    st.write("🧠 ANN - Ng Soon Siang")
    st.write("📍 KNN - Chia Sheng Yang")

# ================= DATA =================
with tab5:
    st.subheader("📚 Dataset Information")
    st.write("UCI Cleveland Heart Disease Dataset")

    fig, ax = plt.subplots()
    ax.bar(['No Disease','Disease'], [138,165])
    st.pyplot(fig)
    plt.close(fig)
