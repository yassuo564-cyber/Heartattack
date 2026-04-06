import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load models
ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

st.sidebar.header("Settings")
page = st.sidebar.selectbox("Page", ["Prediction", "Model Comparison"])
model_choice = st.sidebar.selectbox("Select Model", ["ANN", "KNN"])

if page == "Prediction":
    st.title("Heart Disease Prediction")
    st.write(f"Current model: **{model_choice}**")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Blood Pressure", min_value=50, max_value=300, value=120)
        chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
        fbs = st.selectbox("High Blood Sugar?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Heart Test Result", [0, 1, 2])

    with col2:
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Chest Pain During Exercise?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("Stress Test Value", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Stress Test Slope", [0, 1, 2])
        ca = st.selectbox("Number of Blood Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Blood Disorder Type", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    if st.button("Predict"):
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
            st.error("Prediction: Heart Disease Detected")
        else:
            st.success("Prediction: No Heart Disease")

elif page == "Model Comparison":
    st.title("Model Comparison: ANN vs KNN")

    st.subheader("Performance Metrics")
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'ANN': [0.902, 0.933, 0.875, 0.903],
        'KNN': [0.9016, 0.9333, 0.8750, 0.9032]
    })
    st.table(comparison)

    st.subheader("Performance Chart")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#3498db')
    ax.bar(x + width/2, [0.9016, 0.9333, 0.8750, 0.9032], width, label='KNN', color='#e74c3c')
    ax.set_ylabel('Score')
    ax.set_title('ANN vs KNN Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    st.pyplot(fig)

    st.subheader("Confusion Matrix")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**ANN**")
        fig1, ax1 = plt.subplots()
        cm_ann = np.array([[27, 2], [4, 28]])
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        st.pyplot(fig1)

    with col2:
        st.write("**KNN**")
        fig2, ax2 = plt.subplots()
        cm_knn = np.array([[27, 2], [4, 28]])
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        st.pyplot(fig2)

    st.subheader("Summary")
    st.write("Both ANN and KNN achieved similar performance with accuracy above 90%. Both models had the same confusion matrix results with 27 true negatives, 2 false positives, 4 false negatives and 28 true positives.")
