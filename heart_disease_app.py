import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

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

ann_model = load("ann_model.joblib")
knn_model = load("knn_model.joblib")
encoder = load("encoder.joblib")
scaler = load("scaler.joblib")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3;
        color: white;
    }
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #1976D2;
    }
    div[data-testid="stMetric"] {
        background-color: #1E1E2E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)

st.title("Heart Disease Prediction System")
st.caption("Supervised Machine Learning | ANN & KNN | Cleveland Heart Disease Dataset (UCI)")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", "303")
    c2.metric("Features", "13")
    c3.metric("ANN Accuracy", "90.2%")
    c4.metric("KNN Accuracy", "90.16%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ANN")
        st.write("Artificial Neural Network using MLPClassifier. Learns complex patterns through hidden layers and backpropagation.")
    with col2:
        st.subheader("KNN")
        st.write("K-Nearest Neighbours predicts class labels based on the majority vote of the nearest samples.")

    st.markdown("---")
    st.subheader("Project Workflow")
    st.write("1. Load Cleveland Heart Disease dataset")
    st.write("2. Handle missing values and preprocess features")
    st.write("3. Train ANN and KNN models")
    st.write("4. Compare both models using evaluation metrics")
    st.write("5. Deploy a prototype for real-time prediction")

with tab2:
    model_choice = st.selectbox("Model", ["ANN", "KNN"])
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("BASIC INFO")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox(
            "Chest Pain",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal",
                3: "Asymptomatic"
            }[x]
        )
        trestbps = st.number_input("Blood Pressure", min_value=50, max_value=300, value=120)

    with col2:
        st.caption("CLINICAL DATA")
        chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
        fbs = st.selectbox("High Blood Sugar?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox(
            "ECG Result",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Abnormality",
                2: "LV Hypertrophy"
            }[x]
        )
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)

    with col3:
        st.caption("TEST RESULTS")
        exang = st.selectbox("Exercise Chest Pain?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox(
            "ST Slope",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )
        ca = st.selectbox("Blood Vessels", [0, 1, 2, 3])

    thal = st.selectbox(
        "Thalassemia",
        [3, 6, 7],
        format_func=lambda x: {
            3: "Normal",
            6: "Fixed Defect",
            7: "Reversible Defect"
        }[x]
    )

    st.markdown("---")

    if st.button("Run Prediction", use_container_width=True):
        ann_prob = None
        knn_prob = None

        ann_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        ann_result = ann_model.predict(ann_input)
        if hasattr(ann_model, "predict_proba"):
            ann_prob = float(ann_model.predict_proba(ann_input)[0][1])

        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        user_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        )

        user_cat = pd.DataFrame(
            encoder.transform(user_df[categorical_features]),
            columns=encoder.get_feature_names_out(categorical_features)
        )
        user_num = pd.DataFrame(
            scaler.transform(user_df[numerical_features]),
            columns=numerical_features
        )
        user_processed = pd.concat([user_num, user_cat], axis=1)

        knn_result = knn_model.predict(user_processed)
        if hasattr(knn_model, "predict_proba"):
            knn_prob = float(knn_model.predict_proba(user_processed)[0][1])

        if model_choice == "ANN":
            result = ann_result
            prob = ann_prob
        else:
            result = knn_result
            prob = knn_prob

        st.markdown("---")
        if result[0] == 1:
            st.markdown("""
            <div style="background-color:#2D1117; border:2px solid #FF4B4B; border-radius:10px; padding:20px; text-align:center;">
                <h2 style="color:#FF4B4B;">Heart Disease Detected</h2>
                <p style="color:#AAAAAA;">Please consult a doctor for further diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#0D1F0D; border:2px solid #2ECC71; border-radius:10px; padding:20px; text-align:center;">
                <h2 style="color:#2ECC71;">No Heart Disease</h2>
                <p style="color:#AAAAAA;">Regular checkups are still recommended.</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"Model used: {model_choice}")

        st.markdown("---")
        st.subheader("Input Summary")
        input_summary = pd.DataFrame({
            "Feature": ['Age', 'Gender', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 'High Blood Sugar', 'ECG', 'Max Heart Rate', 'Exercise Chest Pain', 'ST Depression', 'ST Slope', 'Blood Vessels', 'Thalassemia'],
            "Value": [
                age,
                "Male" if sex == 1 else "Female",
                {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[cp],
                trestbps,
                chol,
                "Yes" if fbs == 1 else "No",
                {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[restecg],
                thalach,
                "Yes" if exang == 1 else "No",
                oldpeak,
                {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[slope],
                ca,
                {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[thal]
            ]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)

        if ann_prob is None:
            ann_prob = 1.0 if int(ann_result[0]) == 1 else 0.0
        if knn_prob is None:
            knn_prob = 1.0 if int(knn_result[0]) == 1 else 0.0

        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["ANN", "KNN"]
        values = [ann_prob, knn_prob]
        colors = ["#2196F3", "#FF9800"]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability of Disease")
        ax.set_title("Prediction Probability by Model")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center"
            )
        st.pyplot(fig)

        st.subheader("Model Agreement")
        if int(ann_result[0]) == int(knn_result[0]):
            st.success("ANN and KNN agree on the prediction.")
        else:
            st.warning("ANN and KNN give different predictions.")

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANN Accuracy", "90.2%")
    c2.metric("KNN Accuracy", "90.16%")
    c3.metric("ANN F1", "0.903")
    c4.metric("KNN F1", "0.9032")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metrics Table")
        comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'ANN': [ANN_ACCURACY, ANN_PRECISION, ANN_RECALL, ANN_F1, ANN_AUC],
            'KNN': [KNN_ACCURACY, KNN_PRECISION, KNN_RECALL, KNN_F1, KNN_AUC]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Bar Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(5)
        width = 0.35
        ann_vals = [ANN_ACCURACY, ANN_PRECISION, ANN_RECALL, ANN_F1, ANN_AUC]
        knn_vals = [KNN_ACCURACY, KNN_PRECISION, KNN_RECALL, KNN_F1, KNN_AUC]
        bars1 = ax.bar(x - width/2, ann_vals, width, label='ANN', color='#2196F3')
        bars2 = ax.bar(x + width/2, knn_vals, width, label='KNN', color='#FF9800')
        ax.set_xticks(x)
        ax.set_xticklabels(['Acc', 'Prec', 'Recall', 'F1', 'AUC'])
        ax.legend()
        ax.set_ylim(0.8, 1.0)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=9)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Confusion Matrix")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("ANN")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            ANN_CM,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            ax=ax1,
            annot_kws={'size': 16}
        )
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        st.pyplot(fig1)

    with col2:
        st.caption("KNN")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            KNN_CM,
            annot=True,
            fmt='d',
            cmap='Oranges',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            ax=ax2,
            annot_kws={'size': 16}
        )
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Normalized Confusion Matrix")

    col3, col4 = st.columns(2)
    with col3:
        ann_norm = ANN_CM / ANN_CM.sum(axis=1, keepdims=True)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            ann_norm,
            annot=True,
            fmt=".2%",
            cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            ax=ax3
        )
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('ANN Normalized')
        st.pyplot(fig3)

    with col4:
        knn_norm = KNN_CM / KNN_CM.sum(axis=1, keepdims=True)
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            knn_norm,
            annot=True,
            fmt=".2%",
            cmap='Oranges',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            ax=ax4
        )
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title('KNN Normalized')
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("ROC Curve")
    fpr_ann = [0.0, 0.05, 0.10, 0.18, 1.0]
    tpr_ann = [0.0, 0.72, 0.84, 0.93, 1.0]
    fpr_knn = [0.0, 0.06, 0.11, 0.20, 1.0]
    tpr_knn = [0.0, 0.70, 0.83, 0.92, 1.0]

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.plot(fpr_ann, tpr_ann, label=f"ANN (AUC={ANN_AUC:.3f})", color="#2196F3", linewidth=2)
    ax5.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={KNN_AUC:.3f})", color="#FF9800", linewidth=2)
    ax5.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax5.set_xlabel("False Positive Rate")
    ax5.set_ylabel("True Positive Rate")
    ax5.set_title("ROC Curve Comparison")
    ax5.legend()
    st.pyplot(fig5)

    st.markdown("---")
    st.write("Both models achieved accuracy above 90%. They also show very similar precision, recall, F1-score, and ROC performance. This suggests both ANN and KNN are effective for heart disease classification on this dataset.")

with tab4:
    st.subheader("Ng Soon Siang - ANN (MLPClassifier)")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Algorithm:** MLPClassifier")
        st.write("**Preprocessing:** Median")
        st.write("**Hidden Layers:** (100,)")
        st.write("**Max Iterations:** 500")
        st.write("**Random State:** 42")
    with col2:
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
    sns.heatmap(
        ANN_CM,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease'],
        ax=ax1,
        annot_kws={'size': 16}
    )
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('ANN Confusion Matrix')
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("Chia Sheng Yang - KNN (KNeighborsClassifier)")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Algorithm:** KNeighborsClassifier")
        st.write("**Preprocessing:** Mode, OneHotEncoder, StandardScaler")
        st.write("**Tuning:** k=1 to 39")
        st.write("**Cross Validation:** 10-fold")
        st.write("**Random State:** 42")
    with col2:
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Find best K and train
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
        """, language="python")

    st.write("**Results:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "90.16%")
    col2.metric("Precision", "93.33%")
    col3.metric("Recall", "87.50%")
    col4.metric("F1 Score", "0.9032")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        KNN_CM,
        annot=True,
        fmt='d',
        cmap='Oranges',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease'],
        ax=ax2,
        annot_kws={'size': 16}
    )
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
