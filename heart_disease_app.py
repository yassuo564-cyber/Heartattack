import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #AAAAAA;
        text-align: center;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #AAAAAA;
    }
    .result-box-danger {
        background-color: #2D1117;
        border: 2px solid #FF4B4B;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-box-safe {
        background-color: #0D1F0D;
        border: 2px solid #2ECC71;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .section-divider {
        border-top: 2px solid #333;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# Sidebar
# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="color: #FF4B4B; font-size: 1.5rem; margin-bottom: 0;">Heart Disease</h1>
    <p style="color: #AAAAAA; font-size: 0.9rem;">Prediction System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio("", ["Home", "Prediction", "Model Comparison", "Dataset Info"])

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="padding: 0.5rem 0;">
    <p style="color: #FF4B4B; font-size: 0.8rem; margin-bottom: 0.3rem;">ALGORITHMS</p>
    <p style="color: #FFFFFF; font-size: 0.85rem;">ANN & KNN</p>
    <p style="color: #FF4B4B; font-size: 0.8rem; margin-bottom: 0.3rem;">DATASET</p>
    <p style="color: #FFFFFF; font-size: 0.85rem;">Cleveland Heart Disease</p>
    <p style="color: #FF4B4B; font-size: 0.8rem; margin-bottom: 0.3rem;">SOURCE</p>
    <p style="color: #FFFFFF; font-size: 0.85rem;">UCI Repository</p>
</div>
""", unsafe_allow_html=True)

# ============ HOME ============
if page == "Home":
    st.markdown('<p class="main-header">Heart Disease Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Artificial Neural Network (ANN) and K-Nearest Neighbours (KNN)</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">303</div>
            <div class="metric-label">Patient Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">13</div>
            <div class="metric-label">Clinical Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">90.2%</div>
            <div class="metric-label">ANN Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">90.1%</div>
            <div class="metric-label">KNN Accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ANN (Artificial Neural Network)")
        st.write("MLPClassifier from scikit-learn with 100 hidden neurons. The model learns patterns in the data through backpropagation. It adjusts its weights during training to minimise prediction error.")
    with col2:
        st.markdown("### KNN (K-Nearest Neighbours)")
        st.write("KNeighborsClassifier from scikit-learn. It classifies new data by looking at the k closest training data points and assigning the majority class among those neighbours.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### How to Use")
    st.write("Use the sidebar to navigate between pages. Go to **Prediction** to test the model with patient data. Go to **Model Comparison** to see how ANN and KNN performed.")

# ============ PREDICTION ============
elif page == "Prediction":
    st.markdown('<p class="main-header">Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter patient data below and select a model to predict heart disease</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model", ["ANN", "KNN"])

    st.markdown("### Basic Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x])

    st.markdown("### Clinical Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        trestbps = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=300, value=120)
        fbs = st.selectbox("High Blood Sugar? (>120 mg/dl)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
        restecg = st.selectbox("ECG Result", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
    with col3:
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Chest Pain During Exercise?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("### Additional Tests")
    col1, col2, col3 = st.columns(3)
    with col1:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    with col2:
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    with col3:
        ca = st.selectbox("Blood Vessels (Fluoroscopy)", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.button("Predict", use_container_width=True):
        if model_choice == "ANN":
            user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            result = ann_model.predict(user_data)
        else:
            categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            user_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                   columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            user_cat = pd.DataFrame(encoder.transform(user_df[categorical_features]),
                                    columns=encoder.get_feature_names_out(categorical_features))
            user_num = pd.DataFrame(scaler.transform(user_df[numerical_features]),
                                    columns=numerical_features)
            user_processed = pd.concat([user_num, user_cat], axis=1)
            result = knn_model.predict(user_processed)

        if result[0] == 1:
            st.markdown("""<div class="result-box-danger">
                <h2 style="color: #FF4B4B;">Heart Disease Detected</h2>
                <p style="color: #AAAAAA;">The model predicts that this patient has heart disease. Please consult a doctor for further diagnosis.</p>
                <p style="color: #888;">Model: {}</p>
            </div>""".format(model_choice), unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-box-safe">
                <h2 style="color: #2ECC71;">No Heart Disease</h2>
                <p style="color: #AAAAAA;">The model predicts that this patient does not have heart disease. Regular checkups are still recommended.</p>
                <p style="color: #888;">Model: {}</p>
            </div>""".format(model_choice), unsafe_allow_html=True)

# ============ COMPARISON ============
elif page == "Model Comparison":
    st.markdown('<p class="main-header">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ANN vs KNN Performance Analysis</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ANN Accuracy", "90.2%")
    with col2:
        st.metric("KNN Accuracy", "90.16%")
    with col3:
        st.metric("ANN F1 Score", "0.903")
    with col4:
        st.metric("KNN F1 Score", "0.903")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Performance Metrics")
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'ANN': [0.902, 0.933, 0.875, 0.903],
        'KNN': [0.9016, 0.9333, 0.8750, 0.9032]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Performance Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    x = np.arange(4)
    width = 0.35
    bars1 = ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#FF4B4B')
    bars2 = ax.bar(x + width/2, [0.9016, 0.9333, 0.8750, 0.9032], width, label='KNN', color='#4B9DFF')
    ax.set_ylabel('Score', color='white')
    ax.set_title('ANN vs KNN', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='white')
    ax.tick_params(colors='white')
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color='white')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color='white')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ANN")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        fig1.patch.set_facecolor('#0E1117')
        ax1.set_facecolor('#0E1117')
        cm_ann = np.array([[27, 2], [4, 28]])
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax1, annot_kws={'size': 16})
        ax1.set_xlabel('Predicted', color='white')
        ax1.set_ylabel('Actual', color='white')
        ax1.tick_params(colors='white')
        st.pyplot(fig1)
    with col2:
        st.markdown("#### KNN")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#0E1117')
        ax2.set_facecolor('#0E1117')
        cm_knn = np.array([[27, 2], [4, 28]])
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax2, annot_kws={'size': 16})
        ax2.set_xlabel('Predicted', color='white')
        ax2.set_ylabel('Actual', color='white')
        ax2.tick_params(colors='white')
        st.pyplot(fig2)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Summary")
    st.write("Both ANN and KNN achieved similar performance with accuracy above 90%. Both models correctly identified 28 out of 32 heart disease cases. 4 cases were missed by both models. Neither model is significantly better than the other on this dataset.")

# ============ DATASET INFO ============
elif page == "Dataset Info":
    st.markdown('<p class="main-header">Dataset Information</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Cleveland Heart Disease Dataset from UCI Machine Learning Repository</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Overview")
        st.write("**Source:** UCI Machine Learning Repository")
        st.write("**Author:** Detrano et al. (1989)")
        st.write("**Records:** 303 patients")
        st.write("**Features:** 13 clinical attributes")
        st.write("**Target:** Heart disease (0 = No, 1 = Yes)")
    with col2:
        st.markdown("### Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        bars = ax.bar(['No Disease', 'Disease'], [138, 165], color=['#2ECC71', '#FF4B4B'])
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, str(int(bar.get_height())), ha='center', fontsize=14, color='white')
        st.pyplot(fig)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Feature Description")
    features = pd.DataFrame({
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Description': ['Age in years', 'Gender (1=Male, 0=Female)', 'Chest pain type (0-3)', 'Blood pressure (mm Hg)', 'Cholesterol (mg/dl)', 'Blood sugar >120 (1=Yes, 0=No)', 'ECG results (0-2)', 'Max heart rate', 'Exercise chest pain (1=Yes, 0=No)', 'ST depression value', 'ST segment slope (0-2)', 'Blood vessels count (0-3)', 'Blood disorder (3,6,7)'],
        'Type': ['Numerical', 'Categorical', 'Categorical', 'Numerical', 'Numerical', 'Categorical', 'Categorical', 'Numerical', 'Categorical', 'Numerical', 'Categorical', 'Categorical', 'Categorical']
    })
    st.dataframe(features, use_container_width=True, hide_index=True)
