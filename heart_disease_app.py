from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from train_models import ensure_artifacts


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

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

FEATURE_INFO = {
    "age": {"label": "Age", "kind": "number", "min": 20, "max": 100, "value": 54, "step": 1},
    "sex": {"label": "Sex", "kind": "select", "options": {"Female": 0, "Male": 1}},
    "cp": {
        "label": "Chest Pain Type",
        "kind": "select",
        "options": {
            "Typical angina (1)": 1,
            "Atypical angina (2)": 2,
            "Non-anginal pain (3)": 3,
            "Asymptomatic (4)": 4,
        },
    },
    "trestbps": {
        "label": "Resting Blood Pressure (mmHg)",
        "kind": "number",
        "min": 80,
        "max": 220,
        "value": 130,
        "step": 1,
    },
    "chol": {
        "label": "Serum Cholesterol (mg/dL)",
        "kind": "number",
        "min": 100,
        "max": 620,
        "value": 246,
        "step": 1,
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dL",
        "kind": "select",
        "options": {"No (0)": 0, "Yes (1)": 1},
    },
    "restecg": {
        "label": "Resting ECG Result",
        "kind": "select",
        "options": {
            "Normal (0)": 0,
            "ST-T wave abnormality (1)": 1,
            "Left ventricular hypertrophy (2)": 2,
        },
    },
    "thalach": {
        "label": "Maximum Heart Rate",
        "kind": "number",
        "min": 60,
        "max": 220,
        "value": 150,
        "step": 1,
    },
    "exang": {
        "label": "Exercise Induced Angina",
        "kind": "select",
        "options": {"No (0)": 0, "Yes (1)": 1},
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "kind": "number",
        "min": 0.0,
        "max": 7.0,
        "value": 1.0,
        "step": 0.1,
    },
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "kind": "select",
        "options": {
            "Upsloping (1)": 1,
            "Flat (2)": 2,
            "Downsloping (3)": 3,
        },
    },
    "ca": {
        "label": "Number of Major Vessels",
        "kind": "select",
        "options": {"0": 0, "1": 1, "2": 2, "3": 3},
    },
    "thal": {
        "label": "Thalassemia",
        "kind": "select",
        "options": {
            "Normal (3)": 3,
            "Fixed defect (6)": 6,
            "Reversible defect (7)": 7,
        },
    },
}


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
            radial-gradient(circle at top left, rgba(214, 48, 49, 0.15), transparent 28%),
            radial-gradient(circle at top right, rgba(9, 132, 227, 0.12), transparent 30%),
            linear-gradient(180deg, #fffaf7 0%, #f6f7fb 100%);
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #101828 0%, #1d3557 55%, #457b9d 100%);
        color: white;
        box-shadow: 0 18px 50px rgba(16, 24, 40, 0.18);
    }
    .card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(16, 24, 40, 0.08);
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    }
    .risk-high {
        color: #b42318;
        font-weight: 700;
    }
    .risk-low {
        color: #027a48;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> dict[str, object]:
    ensure_artifacts()
    return {
        "ann": joblib.load(MODEL_DIR / "ann_model.joblib"),
        "knn": joblib.load(MODEL_DIR / "knn_model.joblib"),
    }


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return ensure_artifacts()
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_input_frame(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[values[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def prediction_label(value: int) -> str:
    return "Heart Disease Detected" if int(value) == 1 else "No Heart Disease Detected"


def confidence_class(probability: float) -> str:
    return "risk-high" if probability >= 0.5 else "risk-low"


def plot_confusion_matrix(matrix: list[list[int]], title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="#111827", fontsize=12)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def render_manual_form() -> dict[str, float]:
    values: dict[str, float] = {}
    cols = st.columns(2)
    for idx, feature in enumerate(FEATURE_COLUMNS):
        info = FEATURE_INFO[feature]
        with cols[idx % 2]:
            if info["kind"] == "number":
                values[feature] = st.number_input(
                    info["label"],
                    min_value=info["min"],
                    max_value=info["max"],
                    value=info["value"],
                    step=info["step"],
                )
            else:
                selected_label = st.selectbox(info["label"], list(info["options"].keys()))
                values[feature] = info["options"][selected_label]
    return values


def validate_batch_columns(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    return True, ""


def main() -> None:
    models = load_models()
    metrics = load_metrics()
    best_model_key = max(metrics["models"], key=lambda key: metrics["models"][key]["f1"])

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin-bottom:0.35rem;">Heart Disease Prediction Studio</h1>
            <p style="margin:0;font-size:1.05rem;">
                Streamlit prototype based on ANN and KNN, built to match the assignment rubric:
                preprocessing, model comparison, evaluation metrics, and working prediction demo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        st.markdown(
            f"""
            <div class="card">
                <h4>Dataset Size</h4>
                <h2>{metrics['dataset']['rows']} records</h2>
                <p>Cleveland heart disease dataset with binary target conversion.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_mid:
        st.markdown(
            f"""
            <div class="card">
                <h4>Best Test F1</h4>
                <h2>{metrics['models'][best_model_key]['f1']:.3f}</h2>
                <p>{best_model_key.upper()} performed best on the hold-out test set.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown(
            """
            <div class="card">
                <h4>Rubric Coverage</h4>
                <h2>ANN + KNN + Demo</h2>
                <p>Includes interface, comparison, validation, and batch prediction support.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Project Overview", "Model Comparison", "Single Prediction", "Batch Prediction"]
    )

    with tab1:
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("How this app matches the assignment rubric")
            st.markdown(
                """
                - Identifies a supervised classification problem: heart disease detection.
                - Uses two classification methods from your assignment work: `ANN` and `KNN`.
                - Includes preprocessing for missing values, scaling, and categorical encoding.
                - Compares accuracy, precision, recall, F1 score, and ROC AUC.
                - Provides a usable prototype with single prediction and CSV batch prediction.
                """
            )
            st.subheader("Dataset Summary")
            summary_df = pd.DataFrame(
                {
                    "Metric": [
                        "Rows",
                        "Features",
                        "Positive Class",
                        "Negative Class",
                        "Missing `ca`",
                        "Missing `thal`",
                    ],
                    "Value": [
                        metrics["dataset"]["rows"],
                        len(FEATURE_COLUMNS),
                        metrics["dataset"]["positive_cases"],
                        metrics["dataset"]["negative_cases"],
                        metrics["dataset"]["missing_counts"]["ca"],
                        metrics["dataset"]["missing_counts"]["thal"],
                    ],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Feature Dictionary")
            dictionary_df = pd.DataFrame(
                [{"Feature": col, "Description": FEATURE_INFO[col]["label"]} for col in FEATURE_COLUMNS]
            )
            st.dataframe(dictionary_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("ANN vs KNN Evaluation")
        compare_df = pd.DataFrame(
            [
                {
                    "Model": model_name.upper(),
                    "Accuracy": f"{info['accuracy']:.3f}",
                    "Precision": f"{info['precision']:.3f}",
                    "Recall": f"{info['recall']:.3f}",
                    "F1": f"{info['f1']:.3f}",
                    "ROC AUC": f"{info['roc_auc']:.3f}",
                }
                for model_name, info in metrics["models"].items()
            ]
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(
                plot_confusion_matrix(metrics["models"]["ann"]["confusion_matrix"], "ANN Confusion Matrix"),
                use_container_width=False,
            )
        with col2:
            st.pyplot(
                plot_confusion_matrix(metrics["models"]["knn"]["confusion_matrix"], "KNN Confusion Matrix"),
                use_container_width=False,
            )

        st.caption(
            f"KNN best k from cross-validation: {metrics['models']['knn']['best_k']} | "
            f"ANN hidden layers: {metrics['models']['ann']['hidden_layer_sizes']}"
        )

    with tab3:
        st.subheader("Single Patient Prediction")
        st.caption("Fill in patient details below. Both ANN and KNN will predict the outcome.")

        with st.form("prediction_form"):
            form_values = render_manual_form()
            submitted = st.form_submit_button("Run Prediction", use_container_width=True)

        if submitted:
            input_df = build_input_frame(form_values)
            ann_prob = float(models["ann"].predict_proba(input_df)[0][1])
            knn_prob = float(models["knn"].predict_proba(input_df)[0][1])
            ann_pred = int(models["ann"].predict(input_df)[0])
            knn_pred = int(models["knn"].predict(input_df)[0])
            consensus_prob = (ann_prob + knn_prob) / 2
            consensus_pred = 1 if consensus_prob >= 0.5 else 0

            result_cols = st.columns(3)
            result_payload = [
                ("ANN", ann_pred, ann_prob),
                ("KNN", knn_pred, knn_prob),
                ("Consensus", consensus_pred, consensus_prob),
            ]

            for col, (name, pred, prob) in zip(result_cols, result_payload):
                with col:
                    st.markdown(
                        f"""
                        <div class="card">
                            <h4>{name}</h4>
                            <p class="{confidence_class(prob)}">{prediction_label(pred)}</p>
                            <h2>{prob:.1%}</h2>
                            <p>Estimated probability of heart disease</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("Show Input Data"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Batch Prediction using CSV")
        template_df = pd.DataFrame(
            [
                {
                    "age": 55,
                    "sex": 1,
                    "cp": 2,
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

        uploaded = st.file_uploader("Upload a CSV with all required feature columns", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            valid, message = validate_batch_columns(batch_df)
            if not valid:
                st.error(message)
            else:
                features_df = batch_df[FEATURE_COLUMNS].copy()
                batch_df["ann_probability"] = models["ann"].predict_proba(features_df)[:, 1]
                batch_df["ann_prediction"] = models["ann"].predict(features_df)
                batch_df["knn_probability"] = models["knn"].predict_proba(features_df)[:, 1]
                batch_df["knn_prediction"] = models["knn"].predict(features_df)
                batch_df["consensus_probability"] = (
                    batch_df["ann_probability"] + batch_df["knn_probability"]
                ) / 2
                batch_df["consensus_prediction"] = (
                    batch_df["consensus_probability"] >= 0.5
                ).astype(int)

                st.success("Batch prediction completed successfully.")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "Download Prediction Results",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="heart_predictions_output.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
