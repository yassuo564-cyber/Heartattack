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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 500;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-sub {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        color: #667eea;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .card-text {
        color: #cccccc;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .result-danger {
        background: linear-gradient(135deg, #2d1117 0%, #3d1a1a 100%);
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #0d1f0d 0%, #1a3d1a 100%);
        border: 2px solid #51cf66;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .member-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a4a 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .member-header-knn {
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a4a 100%);
        border-left: 4px solid #FF9800;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

ann_model = load('ann_model.joblib')
knn_model = load('knn_model.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# Header
st.markdown("""
<div class="header-container">
    <p class="header-title">Heart Disease Prediction System</p>
    <p class="header-sub">Supervised Machine Learning | ANN & KNN | Cleveland Heart Disease Dataset (UCI)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info"])

# ============ HOME ============
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.1%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">ANN (Artificial Neural Network)</div>
            <div class="card-text">MLPClassifier from scikit-learn with 100 hidden neurons. The model learns patterns in the data through backpropagation. It adjusts its weights during training to minimise prediction error.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">KNN (K-Nearest Neighbours)</div>
            <div class="card-text">KNeighborsClassifier from scikit-learn. It classifies new data by looking at the k closest training data points and assigning the majority class among those neighbours using Euclidean distance.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("About This Project")
    st.write("This system predicts whether a patient has heart disease based on 13 clinical features. Two supervised machine learning algorithms (ANN and KNN) are trained, evaluated and compared. The models are deployed on this Streamlit web application for easy access.")

    st.markdown("---")

    st.subheader("How to Use")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Step 1</div>
            <div class="card-text">Go to the Prediction tab. Select ANN or KNN model. Enter patient clinical data.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Step 2</div>
            <div class="card-text">Click Run Prediction. The system will predict if the patient has heart disease or not.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Step 3</div>
            <div class="card-text">Go to Model Comparison to see how ANN and KNN performed. View charts and confusion matrices.</div>
        </div>
        """, unsafe_allow_html=True)

# ============ PREDICTION ============
with tab2:
    st.subheader("Select Model and Enter Patient Data")
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
            st.markdown("""
            <div class="result-danger">
                <h2 style="color: #ff6b6b; margin: 0;">Heart Disease Detected</h2>
                <p style="color: #cccccc; margin-top: 0.5rem;">The model predicts that this patient has heart disease. Please consult a doctor for further diagnosis.</p>
                <p style="color: #888888; font-size: 0.85rem;">Model: {}</p>
            </div>
            """.format(model_choice), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <h2 style="color: #51cf66; margin: 0;">No Heart Disease</h2>
                <p style="color: #cccccc; margin-top: 0.5rem;">The model predicts that this patient does not have heart disease. Regular checkups are still recommended.</p>
                <p style="color: #888888; font-size: 0.85rem;">Model: {}</p>
            </div>
            """.format(model_choice), unsafe_allow_html=True)

# ============ COMPARISON ============
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
        bars1 = ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#667eea')
        bars2 = ax.bar(x + width/2, [0.9016, 0.9333, 0.8750, 0.9032], width, label='KNN', color='#FF9800')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
        ax.legend()
        ax.set_ylim(0.8, 1.0)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=8)
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
    st.subheader("Summary")
    st.write("Both ANN and KNN achieved similar performance with accuracy above 90%. Both models correctly identified 28 out of 32 heart disease cases. 4 cases were missed by both models. Neither model showed a clear advantage over the other on this dataset.")

# ============ MEMBER WORK ============
with tab4:
    st.markdown("""
    <div class="member-header">
        <h3 style="color: #667eea; margin: 0;">Ng Soon Siang — ANN (MLPClassifier)</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Algorithm:** Artificial Neural Network")
        st.write("**Library:** MLPClassifier (scikit-learn)")
        st.write("**Preprocessing:** Missing values filled with median")
        st.write("**Model Settings:** 100 hidden neurons, max_iter=500, random_state=42")
        st.write("**Train/Test Split:** 80/20")
    with col2:
        st.write("**Results:**")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "90.2%")
        c2.metric("F1 Score", "0.903")
        c1.metric("Precision", "93.3%")
        c2.metric("Recall", "87.5%")

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

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax1, annot_kws={'size': 16})
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('ANN Confusion Matrix')
    st.pyplot(fig1)

    st.markdown("---")

    st.markdown("""
    <div class="member-header-knn">
        <h3 style="color: #FF9800; margin: 0;">Chia Sheng Yang — KNN (KNeighborsClassifier)</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Algorithm:** K-Nearest Neighbours")
        st.write("**Library:** KNeighborsClassifier (scikit-learn)")
        st.write("**Preprocessing:** Mode, OneHotEncoder, StandardScaler")
        st.write("**Hyperparameter Tuning:** Tested k=1 to 39")
        st.write("**Train/Test Split:** 80/20")
    with col2:
        st.write("**Results:**")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "90.16%")
        c2.metric("F1 Score", "0.903")
        c1.metric("Precision", "93.33%")
        c2.metric("Recall", "87.50%")

    st.write("**Code Overview:**")
    st.code("""
# Load dataset
df = pd.read_csv(file_path, names=columns, na_values='?')

# Handle missing values
for col in ['ca', 'thal']:
    df[col] = df[col].fillna(df[col].mode()[0])

# OneHotEncoder + StandardScaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

# Find best K
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

# Train KNN with best K
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
    """, language="python")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Oranges',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax2, annot_kws={'size': 16})
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('KNN Confusion Matrix')
    st.pyplot(fig2)

# ============ DATASET ============
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overview")
        st.markdown("""
        <div class="card">
            <div class="card-text">
                <b>Source:</b> UCI Machine Learning Repository<br>
                <b>Author:</b> Detrano et al. (1989)<br>
                <b>Records:</b> 303 patients<br>
                <b>Features:</b> 13 clinical attributes<br>
                <b>Target:</b> Heart disease (0 = No, 1 = Yes)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(['No Disease', 'Disease'], [138, 165], color=['#667eea', '#FF9800'])
        ax.set_ylabel('Count')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, str(int(bar.get_height())), ha='center', fontsize=14)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Features")
    features = pd.DataFrame({
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Description': ['Age', 'Gender', 'Chest pain type', 'Blood pressure', 'Cholesterol', 'High blood sugar', 'ECG result', 'Max heart rate', 'Exercise chest pain', 'ST depression', 'ST slope', 'Blood vessels', 'Thalassemia'],
        'Type': ['Num', 'Cat', 'Cat', 'Num', 'Num', 'Cat', 'Cat', 'Num', 'Cat', 'Num', 'Cat', 'Cat', 'Cat']
    })
    st.dataframe(features, use_container_width=True, hide_index=True)
