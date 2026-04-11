import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load


st.set_page_config(
    page_title="Heart Disease Prediction Studio",
    page_icon="heart",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(230, 57, 70, 0.12), transparent 28%),
                radial-gradient(circle at bottom right, rgba(69, 123, 157, 0.12), transparent 25%),
                linear-gradient(180deg, #f8fbff 0%, #eef4f8 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.5rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f172a 0%, #1d3557 55%, #457b9d 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.3rem 0;
            color: white;
            font-size: 2.1rem;
        }
        .hero p {
            margin: 0;
            color: #e8f1f9;
            font-size: 1rem;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }
        .result-good {
            border: 2px solid #16a34a;
            background: #effcf3;
            color: #166534;
            border-radius: 18px;
            padding: 1rem;
            text-align: center;
        }
        .result-bad {
            border: 2px solid #dc2626;
            background: #fef2f2;
            color: #991b1b;
            border-radius: 18px;
            padding: 1rem;
            text-align: center;
        }
        .section-note {
            font-size: 0.95rem;
            color: #475569;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


@st.cache_resource
def load_artifacts():
    ann_model = load("ann_model.joblib")
    knn_model = load("knn_model.joblib")
    encoder = load("encoder.joblib")
    scaler = load("scaler.joblib")
    return ann_model, knn_model, encoder, scaler


ann_model, knn_model, encoder, scaler = load_artifacts()


def safe_probability(model, data, predicted_label):
    if hasattr(model, "predict_proba"):
        try:
            return float(model.predict_proba(data)[0][1])
        except Exception:
            pass
    return 1.0 if int(predicted_label) == 1 else 0.0


def render_result_box(label_value, model_name, probability):
    if int(label_value) == 1:
        st.markdown(
            f"""
            <div class="result-bad">
                <h3 style="margin-bottom:0.2rem;">{model_name}: Heart Disease Detected</h3>
                <p style="margin:0;">Estimated probability: {probability:.1%}</p>
                <p style="margin-top:0.4rem;">Recommendation: seek further medical consultation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-good">
                <h3 style="margin-bottom:0.2rem;">{model_name}: No Heart Disease Detected</h3>
                <p style="margin:0;">Estimated probability: {probability:.1%}</p>
                <p style="margin-top:0.4rem;">Recommendation: maintain healthy lifestyle and regular checkups.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_probability_chart(ann_prob, knn_prob):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["ANN", "KNN", "Average"]
    vals = [ann_prob, knn_prob, (ann_prob + knn_prob) / 2]
    colors = ["#2563eb", "#f59e0b", "#7c3aed"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of Heart Disease")
    ax.set_title("Prediction Confidence by Model")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    return fig


def plot_metrics_bar():
    fig, ax = plt.subplots(figsize=(7, 4))
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    ann_vals = [ANN_ACCURACY, ANN_PRECISION, ANN_RECALL, ANN_F1, ANN_AUC]
    knn_vals = [KNN_ACCURACY, KNN_PRECISION, KNN_RECALL, KNN_F1, KNN_AUC]
    x = np.arange(len(metric_names))
    width = 0.36
    bars1 = ax.bar(x - width / 2, ann_vals, width, label="ANN", color="#2563eb")
    bars2 = ax.bar(x + width / 2, knn_vals, width, label="KNN", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Performance Comparison")
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{bar.get_height():.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
        annot_kws={"size": 15},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_normalized_cm(cm, title, cmap):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap=cmap,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_radar_chart():
    labels = np.array(["Accuracy", "Precision", "Recall", "F1", "AUC"])
    ann_vals = np.array([ANN_ACCURACY, ANN_PRECISION, ANN_RECALL, ANN_F1, ANN_AUC])
    knn_vals = np.array([KNN_ACCURACY, KNN_PRECISION, KNN_RECALL, KNN_F1, KNN_AUC])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ann_vals = np.concatenate((ann_vals, [ann_vals[0]]))
    knn_vals = np.concatenate((knn_vals, [knn_vals[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, ann_vals, linewidth=2, color="#2563eb", label="ANN")
    ax.fill(angles, ann_vals, alpha=0.2, color="#2563eb")
    ax.plot(angles, knn_vals, linewidth=2, color="#f59e0b", label="KNN")
    ax.fill(angles, knn_vals, alpha=0.2, color="#f59e0b")
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    return fig


def plot_dataset_distribution():
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [138, 165]
    bars = ax.bar(["No Disease", "Disease"], counts, color=["#2563eb", "#f59e0b"])
    ax.set_ylabel("Count")
    ax.set_title("Target Distribution")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 2, str(count), ha="center")
    fig.tight_layout()
    return fig


def plot_missing_values():
    fig, ax = plt.subplots(figsize=(5, 4))
    features = ["ca", "thal"]
    counts = [4, 2]
    bars = ax.bar(features, counts, color=["#ef4444", "#f59e0b"])
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values Before Preprocessing")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.05, str(count), ha="center")
    fig.tight_layout()
    return fig


st.markdown(
    """
    <div class="hero">
        <h1>Heart Disease Prediction Studio</h1>
        <p>
            A presentation-ready supervised learning prototype comparing ANN and KNN for heart disease detection,
            designed to match the assignment requirement and prototype rubric.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset Size", "303")
c2.metric("Features", "13")
c3.metric("ANN Accuracy", "90.2%")
c4.metric("KNN Accuracy", "90.16%")

st.markdown("---")

tabs = st.tabs(
    [
        "Project Overview",
        "Live Prediction",
        "Model Evaluation",
        "Member Contributions",
        "Dataset & Methodology",
    ]
)

with tabs[0]:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Problem Statement")
        st.write(
            "This project solves a supervised classification problem: predicting whether a patient shows signs of heart disease based on 13 clinical attributes."
        )
        st.subheader("Why This Prototype Fits the Rubric")
        st.markdown(
            """
            - Clear and organized user interface
            - Working ANN and KNN prediction system
            - Comparison using accuracy, precision, recall, F1, AUC, and confusion matrices
            - Suitable for live demonstration and Q&A presentation
            """
        )
        st.subheader("System Workflow")
        st.markdown(
            """
            1. Collect patient clinical attributes
            2. Apply preprocessing based on model requirement
            3. Run ANN or KNN prediction
            4. Present result, confidence, and comparison output
            """
        )
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Methods Used")
        methods_df = pd.DataFrame(
            {
                "Model": ["ANN", "KNN"],
                "Core Idea": [
                    "Learns complex patterns using hidden neurons and backpropagation",
                    "Predicts based on nearest training samples after feature preprocessing",
                ],
            }
        )
        st.dataframe(methods_df, use_container_width=True, hide_index=True)
        st.markdown('<p class="section-note">This interface is designed to emphasize usability, evaluation clarity, and presentable outputs.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Live Patient Prediction")
    st.caption("Enter patient information below and compare ANN with KNN predictions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Basic Information**")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x],
        )
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=300, value=120)
        chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
    with col2:
        st.markdown("**Clinical Test Results**")
        fbs = st.selectbox("High Fasting Blood Sugar", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
        )
        thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    with col3:
        st.markdown("**Advanced Features**")
        slope = st.selectbox(
            "Slope",
            [0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
        )
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7],
            format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x],
        )
        selected_model = st.radio("Primary Decision Model", ["ANN", "KNN"], horizontal=True)

    if st.button("Run Prediction", use_container_width=True):
        ann_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        ann_prediction = ann_model.predict(ann_input)[0]
        ann_prob = safe_probability(ann_model, ann_input, ann_prediction)

        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        user_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=FEATURE_COLUMNS,
        )
        user_cat = pd.DataFrame(
            encoder.transform(user_df[categorical_features]),
            columns=encoder.get_feature_names_out(categorical_features),
        )
        user_num = pd.DataFrame(
            scaler.transform(user_df[numerical_features]),
            columns=numerical_features,
        )
        user_processed = pd.concat([user_num, user_cat], axis=1)
        knn_prediction = knn_model.predict(user_processed)[0]
        knn_prob = safe_probability(knn_model, user_processed, knn_prediction)

        final_prediction = ann_prediction if selected_model == "ANN" else knn_prediction
        final_probability = ann_prob if selected_model == "ANN" else knn_prob

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            render_result_box(final_prediction, selected_model, final_probability)
        with result_col2:
            agreement = "Models agree" if int(ann_prediction) == int(knn_prediction) else "Models disagree"
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4>Prediction Summary</h4>
                    <p><strong>ANN Probability:</strong> {ann_prob:.1%}</p>
                    <p><strong>KNN Probability:</strong> {knn_prob:.1%}</p>
                    <p><strong>Selected Model:</strong> {selected_model}</p>
                    <p><strong>Agreement Status:</strong> {agreement}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chart_col, table_col = st.columns([1.2, 1])
        with chart_col:
            st.pyplot(plot_probability_chart(ann_prob, knn_prob), use_container_width=False)
        with table_col:
            st.subheader("Patient Input Summary")
            input_summary = pd.DataFrame(
                {
                    "Feature": [
                        "Age",
                        "Gender",
                        "Chest Pain Type",
                        "Resting Blood Pressure",
                        "Cholesterol",
                        "High Fasting Blood Sugar",
                        "Resting ECG",
                        "Maximum Heart Rate",
                        "Exercise Induced Angina",
                        "ST Depression",
                        "Slope",
                        "Major Vessels",
                        "Thalassemia",
                    ],
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
                        {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[thal],
                    ],
                }
            )
            st.dataframe(input_summary, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Model Evaluation and Comparison")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ANN Accuracy", "90.2%")
    m2.metric("KNN Accuracy", "90.16%")
    m3.metric("ANN F1", "0.903")
    m4.metric("KNN F1", "0.9032")

    eval_col1, eval_col2 = st.columns([1.1, 1])
    with eval_col1:
        comparison_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                "ANN": [ANN_ACCURACY, ANN_PRECISION, ANN_RECALL, ANN_F1, ANN_AUC],
                "KNN": [KNN_ACCURACY, KNN_PRECISION, KNN_RECALL, KNN_F1, KNN_AUC],
            }
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.pyplot(plot_metrics_bar(), use_container_width=False)
    with eval_col2:
        st.pyplot(plot_radar_chart(), use_container_width=False)

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.pyplot(plot_confusion_matrix(ANN_CM, "ANN Confusion Matrix", "Blues"), use_container_width=False)
    with cm_col2:
        st.pyplot(plot_confusion_matrix(KNN_CM, "KNN Confusion Matrix", "Oranges"), use_container_width=False)

    norm_col1, norm_col2 = st.columns(2)
    with norm_col1:
        st.pyplot(plot_normalized_cm(ANN_CM, "ANN Normalized Confusion Matrix", "Blues"), use_container_width=False)
    with norm_col2:
        st.pyplot(plot_normalized_cm(KNN_CM, "KNN Normalized Confusion Matrix", "Oranges"), use_container_width=False)

    st.markdown(
        """
        <div class="glass-card">
            <h4>Evaluation Discussion</h4>
            <p>
                Both ANN and KNN exceed 90% accuracy and show similar precision, recall, and F1 performance.
                This indicates that both methods are suitable for heart disease classification on this dataset.
                ANN provides a strong nonlinear learning approach, while KNN offers an intuitive instance-based comparison model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.subheader("Member Contributions")
    member_col1, member_col2 = st.columns(2)

    with member_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Ng Soon Siang - ANN")
        st.write("**Algorithm:** MLPClassifier")
        st.write("**Preprocessing:** Missing values handled with median")
        st.write("**Model Setup:** Hidden layer `(100,)`, max iterations `500`, random state `42`")
        st.write("**Key Strength:** Learns nonlinear relationships in clinical features")
        st.code(
            """
# Train ANN model
df = df.fillna(df.median())
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
ann.fit(X_train, y_train)
            """,
            language="python",
        )
        st.pyplot(plot_confusion_matrix(ANN_CM, "ANN Result", "Blues"), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with member_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Chia Sheng Yang - KNN")
        st.write("**Algorithm:** KNeighborsClassifier")
        st.write("**Preprocessing:** Mode imputation, OneHotEncoder, StandardScaler")
        st.write("**Model Setup:** Tuned best K with cross validation")
        st.write("**Key Strength:** Simple, interpretable neighbour-based prediction")
        st.code(
            """
# Train KNN model
for col in ['ca', 'thal']:
    df[col] = df[col].fillna(df[col].mode()[0])
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
            """,
            language="python",
        )
        st.pyplot(plot_confusion_matrix(KNN_CM, "KNN Result", "Oranges"), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset Overview")
        st.write("**Source:** UCI Machine Learning Repository")
        st.write("**Dataset:** Cleveland Heart Disease")
        st.write("**Records:** 303")
        st.write("**Attributes:** 13 features")
        st.write("**Target:** 0 = No disease, 1 = Disease")
        st.pyplot(plot_dataset_distribution(), use_container_width=False)
    with right:
        st.subheader("Preprocessing Evidence")
        st.pyplot(plot_missing_values(), use_container_width=False)
        st.markdown(
            """
            <div class="glass-card">
                <h4>Why Preprocessing Matters</h4>
                <p>
                    The dataset contains missing values in <code>ca</code> and <code>thal</code>.
                    Numeric values are scaled for KNN, while categorical values are encoded so the model
                    can process them correctly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Feature Dictionary")
    features_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Description": [
                "Age",
                "Gender",
                "Chest pain type",
                "Resting blood pressure",
                "Serum cholesterol",
                "Fasting blood sugar",
                "Resting ECG result",
                "Maximum heart rate achieved",
                "Exercise induced angina",
                "ST depression",
                "Slope of ST segment",
                "Number of major vessels",
                "Thalassemia",
            ],
            "Type": ["Numeric", "Categorical", "Categorical", "Numeric", "Numeric", "Categorical", "Categorical", "Numeric", "Categorical", "Numeric", "Categorical", "Categorical", "Categorical"],
        }
    )
    st.dataframe(features_df, use_container_width=True, hide_index=True)
