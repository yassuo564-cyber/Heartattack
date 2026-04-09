import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

st.title("Heart Disease Prediction System")
st.caption("Supervised Machine Learning | ANN & KNN | Cleveland Heart Disease Dataset (UCI)")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.1%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ANN")
        st.write("Artificial Neural Network using MLPClassifier. Has 100 hidden neurons. Learns through backpropagation.")
    with col2:
        st.subheader("KNN")
        st.write("K-Nearest Neighbours using KNeighborsClassifier. Classifies based on majority vote of nearest neighbours.")

    st.markdown("---")
    st.subheader("How to Use")
    st.write("Click on the **Prediction** tab to test the model. Click **Model Comparison** to view results. Click **Member Work** to see each member's contribution.")

with tab2:
    model_choice = st.selectbox("Model", ["ANN", "KNN"])
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("BASIC INFO")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x])
        trestbps = st.number_input("Blood Pressure", min_value=50, max_value=300, value=120)

    with col2:
        st.caption("CLINICAL DATA")
        chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
        fbs = st.selectbox("High Blood Sugar?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("ECG Result", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)

    with col3:
        st.caption("TEST RESULTS")
        exang = st.selectbox("Exercise Chest Pain?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
        ca = st.selectbox("Blood Vessels", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    st.markdown("---")

    if st.button("Run Prediction", use_container_width=True):
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
        if result[0] == 1:
            st.error("Result: Heart Disease Detected")
            st.write("Please consult a doctor for further diagnosis.")
        else:
            st.success("Result: No Heart Disease")
            st.write("Regular checkups are still recommended.")
        st.caption(f"Model used: {model_choice}")

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANN Accuracy", "90.2%")
    c2.metric("KNN Accuracy", "90.16%")
    c3.metric("ANN F1", "0.903")
    c4.metric("KNN F1", "0.903")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metrics Table")
        comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'ANN': [0.902, 0.933, 0.875, 0.903],
            'KNN': [0.9016, 0.9333, 0.8750, 0.9032]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Bar Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(4)
        width = 0.35
        ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#2196F3')
        ax.bar(x + width/2, [0.9016, 0.9333, 0.8750, 0.9032], width, label='KNN', color='#FF9800')
        ax.set_xticks(x)
        ax.set_xticklabels(['Acc', 'Prec', 'Recall', 'F1'])
        ax.legend()
        ax.set_ylim(0.8, 1.0)
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("ANN")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax1, annot_kws={'size': 16})
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        st.pyplot(fig1)
    with col2:
        st.caption("KNN")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax2, annot_kws={'size': 16})
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        st.pyplot(fig2)

    st.markdown("---")
    st.write("Both models achieved accuracy above 90%. Both correctly identified 28 out of 32 heart disease cases. 4 cases were missed. There is no significant difference between ANN and KNN on this dataset.")

with tab4:
    st.subheader("Ng Soon Siang - ANN (MLPClassifier)")
    st.markdown("---")

    st.write("**Algorithm:** Artificial Neural Network (MLPClassifier)")
    st.write("**Preprocessing:** Missing values filled with median")
    st.write("**Model Settings:** hidden_layer_sizes=(100,), max_iter=500, random_state=42")
    st.write("**Train/Test Split:** 80% training, 20% testing")

    st.write("**Code Overview:**")
    st.code("""
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=column_names, na_values='?')

# Handle missing values
df = df.fillna(df.median())

# Convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ANN
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
ann.fit(X_train, y_train)
y_pred_ann = ann.predict(X_test)
    """, language="python")

    st.write("**Results:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "90.2%")
    col2.metric("Precision", "93.3%")
    col3.metric("Recall", "87.5%")
    col4.metric("F1 Score", "0.903")

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax1, annot_kws={'size': 16})
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('ANN Confusion Matrix')
    st.pyplot(fig1)

    st.markdown("---")
    st.markdown("---")

    st.subheader("Chia Sheng Yang - KNN (KNeighborsClassifier)")
    st.markdown("---")

    st.write("**Algorithm:** K-Nearest Neighbours (KNeighborsClassifier)")
    st.write("**Preprocessing:** Missing values filled with mode, OneHotEncoder, StandardScaler")
    st.write("**Hyperparameter Tuning:** Tested k=1 to 39, selected best k")
    st.write("**Train/Test Split:** 80% training, 20% testing")

    st.write("**Code Overview:**")
    st.code("""
# Load dataset
df = pd.read_csv(file_path, names=columns, na_values='?')

# Handle missing values
for col in ['ca', 'thal']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert target to binary
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# OneHotEncoder + StandardScaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Find best K
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

# Train KNN with best K
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
    """, language="python")

    st.write("**Results:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "90.16%")
    col2.metric("Precision", "93.33%")
    col3.metric("Recall", "87.50%")
    col4.metric("F1 Score", "0.903")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Oranges',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax2, annot_kws={'size': 16})
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('KNN Confusion Matrix')
    st.pyplot(fig2)

with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overview")
        st.write("**Source:** UCI Machine Learning Repository")
        st.write("**Author:** Detrano et al. (1989)")
        st.write("**Records:** 303 patients")
        st.write("**Features:** 13 clinical attributes")
        st.write("**Target:** Heart disease (0 = No, 1 = Yes)")
    with col2:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['No Disease', 'Disease'], [138, 165], color=['#2196F3', '#FF9800'])
        ax.set_ylabel('Count')
        for i, v in enumerate([138, 165]):
            ax.text(i, v + 2, str(v), ha='center', fontsize=14)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Features")
    features = pd.DataFrame({
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Description': ['Age', 'Gender', 'Chest pain type', 'Blood pressure', 'Cholesterol', 'High blood sugar', 'ECG result', 'Max heart rate', 'Exercise chest pain', 'ST depression', 'ST slope', 'Blood vessels', 'Thalassemia'],
        'Type': ['Num', 'Cat', 'Cat', 'Num', 'Num', 'Cat', 'Cat', 'Num', 'Cat', 'Num', 'Cat', 'Cat', 'Cat']
    })
    st.dataframe(features, use_container_width=True, hide_index=True)
