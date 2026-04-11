import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import plotly.express as px

# ===== 页面配置 =====
st.set_page_config(
    page_title="AI Heart Diagnosis",
    layout="wide",
    page_icon="🫀"
)

# ===== 样式（核心）=====
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* 标题 */
.title {
    text-align:center;
    font-size:48px;
    font-weight:800;
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* 卡片 */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    margin-bottom:20px;
}

/* 按钮 */
.stButton>button {
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    border:none;
    border-radius:10px;
    padding:12px;
    color:white;
    font-weight:bold;
}

/* tabs */
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ===== 模型加载 =====
ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# ===== 标题 =====
st.markdown('<div class="title">🫀 AI Heart Diagnosis</div>', unsafe_allow_html=True)
st.caption("ANN vs KNN | Smart Medical Prediction System")

tabs = st.tabs(["🏠 Dashboard", "🧠 Prediction", "📊 Analytics"])

# =======================
# 🏠 Dashboard
# =======================
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.1%")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Dataset Overview")
    fig = px.pie(
        names=["No Disease", "Disease"],
        values=[138,165],
        hole=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 🧠 Prediction
# =======================
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    model_choice = st.radio("Choose Model", ["ANN", "KNN"])

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Gender", [1,0], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest Pain", [0,1,2,3])
        trestbps = st.slider("Blood Pressure", 80, 200, 120)

    with col2:
        chol = st.slider("Cholesterol", 100, 400, 200)
        fbs = st.selectbox("High Sugar", [0,1])
        restecg = st.selectbox("ECG", [0,1,2])
        thalach = st.slider("Max Heart Rate", 80, 200, 150)

    with col3:
        exang = st.selectbox("Exercise Pain", [0,1])
        oldpeak = st.slider("ST Depression", 0.0, 5.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("Vessels", [0,1,2,3])

    thal = st.selectbox("Thalassemia", [3,6,7])

    if st.button("🚀 Run AI Prediction", use_container_width=True):

        user = [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]

        if model_choice == "ANN":
            result = ann_model.predict([user])
        else:
            df = pd.DataFrame([user], columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                              'thalach','exang','oldpeak','slope','ca','thal'])

            cat = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
            num = ['age','trestbps','chol','thalach','oldpeak']

            df_cat = pd.DataFrame(encoder.transform(df[cat]))
            df_num = pd.DataFrame(scaler.transform(df[num]))

            final = pd.concat([df_num, df_cat], axis=1)
            result = knn_model.predict(final)

        st.markdown("---")

        if result[0] == 1:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#ff416c,#ff4b2b);
                        padding:30px;border-radius:20px;text-align:center;">
                <h2>⚠️ High Risk</h2>
                <p>Consult doctor immediately</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#00c853,#64dd17);
                        padding:30px;border-radius:20px;text-align:center;">
                <h2>✅ Healthy</h2>
                <p>No heart disease detected</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 📊 Analytics
# =======================
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Model Comparison")

    df = pd.DataFrame({
        "Metric": ["Accuracy","Precision","Recall","F1"],
        "ANN": [0.902,0.933,0.875,0.903],
        "KNN": [0.9016,0.9333,0.875,0.9032]
    })

    fig = px.bar(df, x="Metric", y=["ANN","KNN"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
