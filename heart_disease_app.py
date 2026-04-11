import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# ====== 页面配置 ======
st.set_page_config(
    page_title="🔥 Smart ML Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ====== 自定义CSS（关键！让它变好看）======
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main {
    background: transparent;
}

h1 {
    color: #ffffff;
    text-align: center;
}

.card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ====== 标题 ======
st.markdown("<h1>🚀 Smart ML Dashboard</h1>", unsafe_allow_html=True)

# ====== 上传数据 ======
st.markdown('<div class="card">', unsafe_allow_html=True)
file = st.file_uploader("📁 Upload your dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("### 👀 Data Preview")
    st.dataframe(df)

    # ====== 图表 ======
    st.write("### 📊 Visualization")
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[-1])
    st.plotly_chart(fig, use_container_width=True)

    # ====== KNN ======
    st.write("### 🤖 KNN Prediction")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    k = st.slider("Select K value", 1, 10, 3)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        pred = model.predict([input_data])
        st.success(f"Prediction: {pred[0]}")

st.markdown('</div>', unsafe_allow_html=True)

# ====== Footer ======
st.markdown("---")
st.markdown("✨ Built with Streamlit | Designed by YOU 😎")
