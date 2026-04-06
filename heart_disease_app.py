import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load models
ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Comparison", "About Dataset"])
st.sidebar.markdown("---")
st.sidebar.write("**Project:** Heart Disease Prediction")
st.sidebar.write("**Algorithms:** ANN & KNN")
st.sidebar.write("**Dataset:** Cleveland Heart Disease (UCI)")

# ============ HOME PAGE ============
if page == "Home":
    st.title("Heart Disease Prediction System")
    st.write("A Machine Learning Approach Using ANN and KNN")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "303 patients")
    with col2:
        st.metric("Features", "13 clinical")
    with col3:
        st.metric("Best Accuracy", "90.2%")

    st.markdown("---")
    st.subheader("About This Project")
    st.write("This system predicts whether a patient has heart disease based on clinical test results. Two machine learning algorithms are implemented and compared:")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**ANN (Artificial Neural Network)**\n\nMLPClassifier with 100 hidden neurons. Learns patterns through backpropagation.")
    with col2:
        st.info("**KNN (K-Nearest Neighbours)**\n\nClassifies based on majority vote of nearest neighbours using Euclidean distance.")

    st.markdown("---")
    st.subheader("How to Use")
    st.write("1. Go to **Prediction** page to input patient data and get a prediction.")
    st.write("2. Go to **Model Comparison** page to see how ANN and KNN compare.")
    st.write("3. Go to **About Dataset** page to learn about the dataset used.")

# ============ PREDICTION PAGE ============
elif page == "Prediction":
    st.title("Heart Disease Prediction")
    st.markdown("---")

    model_choice = st.selectbox("Select Model", ["ANN", "KNN"])

    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {0: "0 - Typical Angina", 1: "1 - Atypical Angina", 2: "2 - Non-anginal Pain", 3: "3 - Asymptomatic"}[x])
        trestbps = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=300, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
        fbs = st.selectbox("High Blood Sugar? (>120 mg/dl)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Heart Test Result (ECG)", [0, 1, 2], format_func=lambda x: {0: "0 - Normal", 1: "1 - ST-T Abnormality", 2: "2 - Left Ventricular Hypertrophy"}[x])

    with col2:
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Chest Pain During Exercise?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("Stress Test Value (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Stress Test Slope", [0, 1, 2], format_func=lambda x: {0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"}[x])
        ca = st.selectbox("Number of Blood Vessels (Fluoroscopy)", [0, 1, 2, 3])
        thal = st.selectbox("Blood Disorder Type (Thalassemia)", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    st.markdown("---")
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

        st.markdown("---")
        st.subheader("Result")
        if result[0] == 1:
            st.error("Heart Disease Detected")
            st.write("The model predicts that this patient **has heart disease**. Please consult a doctor for further diagnosis.")
        else:
            st.success("No Heart Disease")
            st.write("The model predicts that this patient **does not have heart disease**. However, regular checkups are still recommended.")
        st.info(f"Model used: **{model_choice}**")

# ============ COMPARISON PAGE ============
elif page == "Model Comparison":
    st.title("Model Comparison: ANN vs KNN")
    st.markdown("---")

    st.subheader("Performance Metrics")
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'ANN': [0.902, 0.933, 0.875, 0.903],
        'KNN': [0.9016, 0.9333, 0.8750, 0.9032]
    })
    st.table(comparison)

    st.subheader("Performance Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4)
    width = 0.35
    bars1 = ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#3498db')
    bars2 = ax.bar(x + width/2, [0.9016, 0.9333, 0.8750, 0.9032], width, label='KNN', color='#e74c3c')
    ax.set_ylabel('Score')
    ax.set_title('ANN vs KNN Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Confusion Matrix")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**ANN**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        cm_ann = np.array([[27, 2], [4, 28]])
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax1, annot_kws={'size': 16})
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title('ANN Confusion Matrix')
        st.pyplot(fig1)

    with col2:
        st.write("**KNN**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        cm_knn = np.array([[27, 2], [4, 28]])
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax2, annot_kws={'size': 16})
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('KNN Confusion Matrix')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ANN Accuracy", "90.2%")
        st.metric("ANN F1 Score", "0.903")
    with col2:
        st.metric("KNN Accuracy", "90.16%")
        st.metric("KNN F1 Score", "0.9032")

    st.write("Both ANN and KNN achieved similar performance with accuracy above 90%. Both models correctly identified 28 out of 32 heart disease cases. 4 cases were missed by both models.")

# ============ ABOUT DATASET PAGE ============
elif page == "About Dataset":
    st.title("About the Dataset")
    st.markdown("---")

    st.subheader("Cleveland Heart Disease Dataset")
    st.write("**Source:** UCI Machine Learning Repository")
    st.write("**Created by:** Detrano et al. (1989)")
    st.write("**Total records:** 303 patients")
    st.write("**Features:** 13 clinical attributes")
    st.write("**Target:** Heart disease presence (0 = No, 1 = Yes)")

    st.markdown("---")
    st.subheader("Feature Description")
    features = pd.DataFrame({
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Description': ['Age in years', 'Gender (1=Male, 0=Female)', 'Chest pain type (0-3)', 'Blood pressure (mm Hg)', 'Cholesterol (mg/dl)', 'Blood sugar >120 (1=Yes, 0=No)', 'ECG results (0-2)', 'Max heart rate', 'Chest pain during exercise (1=Yes, 0=No)', 'ST depression value', 'ST segment slope (0-2)', 'Number of blood vessels (0-3)', 'Blood disorder type (3,6,7)'],
        'Type': ['Numerical', 'Categorical', 'Categorical', 'Numerical', 'Numerical', 'Categorical', 'Categorical', 'Numerical', 'Categorical', 'Numerical', 'Categorical', 'Categorical', 'Categorical']
    })
    st.table(features)

    st.markdown("---")
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['No Disease (0)', 'Disease (1)'], [138, 165], color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Count')
    ax.set_title('Target Variable Distribution')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, str(int(bar.get_height())), ha='center', fontsize=12)
    st.pyplot(fig)
