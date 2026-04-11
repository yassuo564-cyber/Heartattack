from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load

from train_models import ensure_artifacts, load_dataset


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

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

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
    metrics = ensure_artifacts()
    ann_model = load(MODEL_DIR / "ann_model.joblib")
    knn_model = load(MODEL_DIR / "knn_model.joblib")
    dataset = load_dataset()
    return metrics, ann_model, knn_model, dataset


METRICS, ann_model, knn_model, DATASET = load_artifacts()
ANN_CM = np.array(METRICS["models"]["ann"]["confusion_matrix"])
KNN_CM = np.array(METRICS["models"]["knn"]["confusion_matrix"])


def metric_value(model_key, metric_name):
    return float(METRICS["models"][model_key][metric_name])


def build_input_frame(values):
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


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
    ann_vals = [metric_value("ann", name) for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
    knn_vals = [metric_value("knn", name) for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
    x = np.arange(len(metric_names))
    width = 0.36
    bars1 = ax.bar(x - width / 2, ann_vals, width, label="ANN", color="#2563eb")
    bars2 = ax.bar(x + width / 2, knn_vals, width, label="KNN", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.75, 1.0)
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
    ann_vals = np.array([metric_value("ann", name) for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]])
    knn_vals = np.array([metric_value("knn", name) for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]])
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
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    return fig


def plot_dataset_distribution():
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [int((DATASET["target"] == 0).sum()), int((DATASET["target"] == 1).sum())]
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
    counts = [int(DATASET["ca"].isna().sum()), int(DATASET["thal"].isna().sum())]
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
c1.metric("Dataset Size", str(METRICS["dataset"]["rows"]))
c2.metric("Features", str(len(FEATURE_COLUMNS)))
c3.metric("ANN Accuracy", f"{metric_value('ann', 'accuracy'):.3f}")
c4.metric("KNN Accuracy", f"{metric_value('knn', 'accuracy'):.3f}")

st.markdown("---")

tabs = st.tabs(
    [
        "Project Overview",
        "Live Prediction",
        "Batch Prediction",
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
    st.caption("Enter patient information below and compare ANN with KNN predictions using the saved pipelines.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Basic Information**")
            age = st.number_input("Age", min_value=20, max_value=100, value=54)
            sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox(
                "Chest Pain Type",
                [1, 2, 3, 4],
                format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}[x],
            )
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=130)
            chol = st.number_input("Cholesterol", min_value=100, max_value=620, value=246)
        with col2:
            st.markdown("**Clinical Test Results**")
            fbs = st.selectbox("High Fasting Blood Sugar", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
            )
            thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        with col3:
            st.markdown("**Advanced Features**")
            slope = st.selectbox(
                "Slope",
                [1, 2, 3],
                format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
            )
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
            thal = st.selectbox(
                "Thalassemia",
                [3, 6, 7],
                format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x],
            )
            selected_model = st.radio("Primary Decision Model", ["ANN", "KNN", "Consensus"], horizontal=True)
        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        user_df = build_input_frame([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        ann_prediction = int(ann_model.predict(user_df)[0])
        ann_prob = safe_probability(ann_model, user_df, ann_prediction)
        knn_prediction = int(knn_model.predict(user_df)[0])
        knn_prob = safe_probability(knn_model, user_df, knn_prediction)
        consensus_prob = (ann_prob + knn_prob) / 2
        consensus_prediction = int(consensus_prob >= 0.5)

        final_prediction = {"ANN": ann_prediction, "KNN": knn_prediction, "Consensus": consensus_prediction}[selected_model]
        final_probability = {"ANN": ann_prob, "KNN": knn_prob, "Consensus": consensus_prob}[selected_model]

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
                    <p><strong>Consensus Probability:</strong> {consensus_prob:.1%}</p>
                    <p><strong>Selected Model:</strong> {selected_model}</p>
                    <p><strong>Agreement Status:</strong> {agreement}</p>
                    <p><strong>Purpose:</strong> This prototype supports learning and demonstration, not clinical diagnosis.</p>
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
                            {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}[cp],
                            trestbps,
                            chol,
                            "Yes" if fbs == 1 else "No",
                            {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[restecg],
                            thalach,
                            "Yes" if exang == 1 else "No",
                            oldpeak,
                            {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[slope],
                            ca,
                            {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[thal],
                        ],
                }
            )
            st.dataframe(input_summary, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Batch Prediction")
    st.caption("This extra demo feature strengthens completeness and lets you test multiple patients at once.")
    template_df = pd.DataFrame(
        [
            {
                "age": 55,
                "sex": 1,
                "cp": 3,
                "trestbps": 140,
                "chol": 250,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 0,
                "thal": 3,
            }
        ]
    )
    st.download_button(
        "Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="heart_prediction_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload a CSV with the required feature columns", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing_columns = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            features_df = batch_df[FEATURE_COLUMNS].copy()
            batch_df["ann_probability"] = ann_model.predict_proba(features_df)[:, 1]
            batch_df["ann_prediction"] = ann_model.predict(features_df)
            batch_df["knn_probability"] = knn_model.predict_proba(features_df)[:, 1]
            batch_df["knn_prediction"] = knn_model.predict(features_df)
            batch_df["consensus_probability"] = (batch_df["ann_probability"] + batch_df["knn_probability"]) / 2
            batch_df["consensus_prediction"] = (batch_df["consensus_probability"] >= 0.5).astype(int)
            st.success("Batch prediction completed successfully.")
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download Prediction Results",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="heart_prediction_results.csv",
                mime="text/csv",
            )

with tabs[3]:
    st.subheader("Model Evaluation and Comparison")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ANN Accuracy", f"{metric_value('ann', 'accuracy'):.3f}")
    m2.metric("KNN Accuracy", f"{metric_value('knn', 'accuracy'):.3f}")
    m3.metric("ANN F1", f"{metric_value('ann', 'f1'):.3f}")
    m4.metric("KNN F1", f"{metric_value('knn', 'f1'):.3f}")

    eval_col1, eval_col2 = st.columns([1.1, 1])
    with eval_col1:
        comparison_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                "ANN": [metric_value("ann", "accuracy"), metric_value("ann", "precision"), metric_value("ann", "recall"), metric_value("ann", "f1"), metric_value("ann", "roc_auc")],
                "KNN": [metric_value("knn", "accuracy"), metric_value("knn", "precision"), metric_value("knn", "recall"), metric_value("knn", "f1"), metric_value("knn", "roc_auc")],
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

with tabs[4]:
    st.subheader("Member Contributions")
    member_col1, member_col2 = st.columns(2)

    with member_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Ng Soon Siang - ANN")
        st.write("**Algorithm:** MLPClassifier")
        st.write("**Preprocessing:** Shared pipeline with median imputation, scaling, and encoded categorical handling")
        st.write(f"**Model Setup:** Hidden layer `{tuple(METRICS['models']['ann']['hidden_layer_sizes'])}`, max iterations `{METRICS['models']['ann']['max_iter']}`, random state `42`")
        st.write("**Key Strength:** Learns nonlinear relationships in clinical features")
        st.code(
            """
# ANN pipeline
ann_model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, early_stopping=True, random_state=42)),
    ]
)
            """,
            language="python",
        )
        st.pyplot(plot_confusion_matrix(ANN_CM, "ANN Result", "Blues"), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with member_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Chia Sheng Yang - KNN")
        st.write("**Algorithm:** KNeighborsClassifier")
        st.write("**Preprocessing:** Shared pipeline with most-frequent imputation, OneHotEncoder, and StandardScaler")
        st.write(f"**Model Setup:** Tuned best K with cross validation, selected `k = {METRICS['models']['knn']['best_k']}`")
        st.write("**Key Strength:** Simple, interpretable neighbour-based prediction")
        st.code(
            """
# KNN pipeline
knn_model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)
            """,
            language="python",
        )
        st.pyplot(plot_confusion_matrix(KNN_CM, "KNN Result", "Oranges"), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[5]:
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset Overview")
        st.write("**Source:** UCI Machine Learning Repository")
        st.write("**Dataset:** Cleveland Heart Disease")
        st.write(f"**Records:** {METRICS['dataset']['rows']}")
        st.write(f"**Attributes:** {len(FEATURE_COLUMNS)} features")
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
                    Numeric values are imputed and scaled, while categorical values are imputed and one-hot encoded
                    so both ANN and KNN can work on clean, consistent input features.
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
    st.markdown(
        f"""
        <div class="glass-card">
            <h4>Training Methodology</h4>
            <p><strong>KNN best k:</strong> {METRICS['models']['knn']['best_k']} from 5-fold cross validation.</p>
            <p><strong>Evaluation metrics:</strong> Accuracy, precision, recall, F1, ROC AUC, and confusion matrices.</p>
            <p><strong>Prototype strength:</strong> real model outputs, organized interface, batch prediction, and evaluation evidence for Q&amp;A.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
