import streamlit as st
from joblib import load
import numpy as np

# Load the model
model = load('ann_model.joblib')

st.title("Heart Disease Prediction (ANN)")

# User input
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=300, value=120)
chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

if st.button("Predict"):
    user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    result = model.predict(user_data)
    if result[0] == 1:
        st.error("Prediction: Heart Disease Detected")
    else:
        st.success("Prediction: No Heart Disease")