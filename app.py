import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

try:
    import shap
except Exception:
    shap = None

MODEL_PATH = "best_model.pkl"
ENCODER_PATHS = {
    "workclass": "workclass_encoder.pkl",
    "occupation": "occupation_encoder.pkl",
    "relationship": "relationship_encoder.pkl",
    "race": "race_encoder.pkl",
    "gender": "gender_encoder.pkl",
    "native-country": "native-country_encoder.pkl",
    "marital-status": "marital-status_encoder.pkl",
}
DATA_PATH = "adult.csv"
HISTORY_FILE = "prediction_history.csv"
FEATURE_COLUMNS = [
    "age",
    "workclass",
    "educational-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
EDUCATION_MAPPING = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource
def load_encoders():
    return {name: joblib.load(path) for name, path in ENCODER_PATHS.items()}


@st.cache_data
def load_dataset(path: str):
    return pd.read_csv(path)


def style_app():
    st.markdown(
        """
        <style>
            :root { color-scheme: light; }
            .stApp { background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%); color: #0f172a; }
            .block-container { padding-top: 1rem; padding-bottom: 3rem; }
            .stSidebar { background: linear-gradient(180deg, #111827 0%, #1f2937 100%); }
            .stSidebar .stMarkdown, .stSidebar .stTextInput label, .stSidebar .stSelectbox label, .stSidebar .stRadio div {
                color: white !important;
            }
            .stButton > button {
                background: linear-gradient(135deg, #2563eb 0%, #4338ca 100%);
                color: white;
                border: none;
                border-radius: 999px;
                padding: 0.7rem 1.1rem;
                font-weight: 600;
            }
            .header-title {
                font-size: 2.2rem;
                font-weight: 700;
                margin-top: 1rem;
                margin-bottom: 0.2rem;
                color: #0f172a;
            }
            .header-subtitle {
                font-size: 1rem;
                color: #475569;
                margin-bottom: 1.2rem;
            }
            .hero-card {
                padding: 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
                border: 1px solid #dbeafe;
                box-shadow: 0 20px 50px rgba(37, 99, 235, 0.08);
                margin-bottom: 1.2rem;
            }
            .card {
                padding: 1.1rem;
                border-radius: 18px;
                background: white;
                border: 1px solid #e2e8f0;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
                margin-bottom: 1rem;
            }
            .feature-card {
                padding: 1rem;
                border-radius: 16px;
                background: white;
                border: 1px solid #e2e8f0;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
                min-height: 110px;
            }
            .result-card {
                padding: 1.3rem;
                border-radius: 20px;
                background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                border: 1px solid #bbf7d0;
                margin-bottom: 1rem;
            }
            .result-title { font-size: 0.95rem; color: #166534; font-weight: 700; }
            .result-value { font-size: 1.8rem; font-weight: 800; color: #14532d; margin-top: 0.3rem; }
            .result-subtitle { font-size: 0.9rem; color: #52525b; margin-top: 0.35rem; }
            .logo-text { font-size: 1.1rem; font-weight: 700; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_label_encode(series: pd.Series, encoder):
    label_to_int = {value: index for index, value in enumerate(encoder.classes_)}
    encoded = series.map(label_to_int)
    if encoded.isna().any():
        encoded = encoded.fillna(-1)
    return encoded.astype(int)


def encode_features(df: pd.DataFrame, encoders: dict):
    encoded = df.copy()
    for column, encoder in encoders.items():
        if column in encoded.columns:
            encoded[column] = safe_label_encode(encoded[column], encoder)
    return encoded


def calculate_importance(model, X: pd.DataFrame, y: pd.Series):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return pd.Series(values, index=X.columns).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        values = abs(model.coef_.ravel())
        return pd.Series(values, index=X.columns).sort_values(ascending=False)
    importance = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    return pd.Series(importance.importances_mean, index=X.columns).sort_values(ascending=False)


def format_probability(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def save_history(entry: dict):
    history_df = load_history()
    history_df = pd.concat([history_df, pd.DataFrame([entry])], ignore_index=True)
    history_df.to_csv(HISTORY_FILE, index=False)


def load_history():
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        expected_columns = [
            "timestamp",
            "age",
            "workclass",
            "occupation",
            "hours-per-week",
            "prediction",
            "confidence",
        ]
        for column in expected_columns:
            if column not in history_df.columns:
                history_df[column] = ""
        return history_df[expected_columns]
    return pd.DataFrame(columns=["timestamp", "age", "workclass", "occupation", "hours-per-week", "prediction", "confidence"])


def create_pdf_report(input_df: pd.DataFrame, prediction: str, confidence: float, factor_frame: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Employee Salary Prediction Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.now().isoformat()}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Employee Details", ln=True)
    pdf.set_font("Arial", size=10)
    for col, value in input_df.iloc[0].items():
        pdf.cell(0, 6, f"{col}: {value}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Prediction", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Income Band: {prediction}", ln=True)
    pdf.cell(0, 6, f"Confidence: {format_probability(confidence)}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Explanation", ln=True)
    pdf.set_font("Arial", size=10)
    explanation_text = "This profile appears to align with the selected income band based on education, role pattern, and work intensity."
    pdf.multi_cell(0, 6, explanation_text)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendations", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, "Use this result as decision support, not a final compensation decision. Review qualifications and market benchmarks for confirmation.")
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Top Influencing Factors", ln=True)
    pdf.set_font("Arial", size=10)
    for _, row in factor_frame.head(6).iterrows():
        pdf.cell(0, 6, f"{row['Feature']}: {row['Importance']:.4f}", ln=True)

    return pdf.output(dest="S").encode("latin-1")


def build_prediction_explanation(prediction, factor_scores, confidence):
    prediction_value = str(prediction).strip()
    is_high_income = prediction_value == ">50K"
    top_factors = [item for item in factor_scores if item[1] > 0][:3]
    if not top_factors:
        top_factors = [("education", 0.25), ("occupation", 0.2)]

    factor_text = ", ".join(f"{feature} ({impact:.2f})" for feature, impact in top_factors)
    if is_high_income:
        explanation = f"This profile appears more likely to fall into the higher-income bracket because the strongest signals are {factor_text}."
    else:
        explanation = f"This profile leans toward the lower-income bracket because the strongest signals are {factor_text}."

    if confidence is not None and confidence < 0.8:
        explanation += " The model confidence is moderate, so treat this as a screening signal rather than a final assessment."
    else:
        explanation += " The confidence level is strong enough to support a preliminary review."

    return explanation


def build_recommendations(prediction, confidence):
    prediction_value = str(prediction).strip()
    recommendations = []
    if prediction_value == ">50K":
        recommendations.append("This profile appears aligned with higher earning potential, which can support conversations about senior or specialist roles.")
    else:
        recommendations.append("This profile suggests a lower-income bracket, so it is worth reviewing role scope, education level, and experience before making a decision.")

    if confidence is not None and confidence < 0.8:
        recommendations.append("Confidence is moderate, so pair this assessment with interview context and market benchmarks.")
    else:
        recommendations.append("Confidence is strong enough to make this a useful first-pass screening signal.")

    recommendations.append("Use this result as decision support rather than a standalone hiring or compensation decision.")
    return recommendations


def generate_factor_summary(employee_row: pd.Series, importance: pd.Series):
    factors = []
    importance_values = importance.head(8)
    for feature, importance_score in importance_values.items():
        if feature == "educational-num":
            value = employee_row.get("educational-num", 0)
            if value >= 13:
                factors.append(("education", importance_score * 1.25))
            elif value >= 10:
                factors.append(("education", importance_score * 1.0))
        elif feature == "hours-per-week":
            value = employee_row.get("hours-per-week", 0)
            if value >= 40:
                factors.append(("hours-per-week", importance_score * 1.15))
            elif value >= 30:
                factors.append(("hours-per-week", importance_score * 0.85))
        elif feature == "capital-gain":
            value = employee_row.get("capital-gain", 0)
            if value > 0:
                factors.append(("capital-gain", importance_score * 1.2))
        elif feature == "capital-loss":
            value = employee_row.get("capital-loss", 0)
            if value > 0:
                factors.append(("capital-loss", importance_score * 0.9))
        elif feature == "occupation":
            value = employee_row.get("occupation", "")
            if value in {"Exec-managerial", "Prof-specialty", "Sales"}:
                factors.append(("occupation", importance_score * 1.1))
        elif feature == "marital-status":
            value = employee_row.get("marital-status", "")
            if value == "Married-civ-spouse":
                factors.append(("marital-status", importance_score * 0.8))
        elif feature == "relationship":
            value = employee_row.get("relationship", "")
            if value in {"Husband", "Wife"}:
                factors.append(("relationship", importance_score * 0.75))
        elif feature == "age":
            value = employee_row.get("age", 0)
            if value >= 35:
                factors.append(("age", importance_score * 0.95))
        else:
            factors.append((feature, importance_score * 0.8))

    if not factors:
        factors = [("education", 0.3), ("occupation", 0.2), ("hours-per-week", 0.15)]

    return sorted(factors, key=lambda item: item[1], reverse=True)


def validate_batch_dataframe(batch_df: pd.DataFrame, reference_df: pd.DataFrame | None = None):
    required_columns = [
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]
    missing_columns = sorted(set(required_columns) - set(batch_df.columns))

    if reference_df is None:
        reference_df = load_dataset(DATA_PATH)

    reference_categories = {
        "workclass": set(reference_df["workclass"].dropna().astype(str).str.strip().tolist()),
        "education": set(reference_df["education"].dropna().astype(str).str.strip().tolist()),
        "marital-status": set(reference_df["marital-status"].dropna().astype(str).str.strip().tolist()),
        "occupation": set(reference_df["occupation"].dropna().astype(str).str.strip().tolist()),
        "relationship": set(reference_df["relationship"].dropna().astype(str).str.strip().tolist()),
        "race": set(reference_df["race"].dropna().astype(str).str.strip().tolist()),
        "gender": set(reference_df["gender"].dropna().astype(str).str.strip().tolist()),
        "native-country": set(reference_df["native-country"].dropna().astype(str).str.strip().tolist()),
    }

    valid_rows = []
    invalid_rows = []
    for index, row in batch_df.iterrows():
        issues = []
        row_data = row.to_dict()
        for column in required_columns:
            value = row_data.get(column)
            if pd.isna(value) or str(value).strip() == "":
                issues.append(f"{column}_missing")

        try:
            age_value = float(row_data.get("age", 0))
            if age_value < 18 or age_value > 100:
                issues.append("age_out_of_range")
        except (TypeError, ValueError):
            issues.append("age_invalid")

        try:
            hours_value = float(row_data.get("hours-per-week", 0))
            if hours_value < 1 or hours_value > 99:
                issues.append("hours_out_of_range")
        except (TypeError, ValueError):
            issues.append("hours_invalid")

        try:
            capital_gain = float(row_data.get("capital-gain", 0))
            capital_loss = float(row_data.get("capital-loss", 0))
            if capital_gain < 0 or capital_loss < 0:
                issues.append("capital_values_invalid")
        except (TypeError, ValueError):
            issues.append("capital_values_invalid")

        education_value = str(row_data.get("education", "")).strip()
        if education_value not in reference_categories["education"]:
            issues.append("education_invalid")

        for categorical_column in ["workclass", "marital-status", "occupation", "relationship", "race", "gender", "native-country"]:
            value = str(row_data.get(categorical_column, "")).strip()
            if value and value not in reference_categories[categorical_column]:
                issues.append(f"{categorical_column}_invalid")

        if issues:
            invalid_rows.append({
                "row_index": index,
                "reason": issues[0],
                "details": issues,
                "data": row_data,
            })
            continue

        normalized_row = {
            "age": int(float(row_data["age"])),
            "workclass": row_data["workclass"],
            "educational-num": EDUCATION_MAPPING.get(education_value, -1),
            "marital-status": row_data["marital-status"],
            "occupation": row_data["occupation"],
            "relationship": row_data["relationship"],
            "race": row_data["race"],
            "gender": row_data["gender"],
            "capital-gain": float(row_data["capital-gain"]),
            "capital-loss": float(row_data["capital-loss"]),
            "hours-per-week": float(row_data["hours-per-week"]),
            "native-country": row_data["native-country"],
            "__row_index__": index,
        }
        valid_rows.append(normalized_row)

    return {
        "missing_columns": missing_columns,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }


def display_home(df: pd.DataFrame, importance: pd.Series):
    st.markdown("<div class='header-title'>🏠 Employee Salary Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>A polished, recruiter-friendly decision-support experience for salary band screening.</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='hero-card'>
            <h3 style='margin:0 0 0.4rem 0;'>Use the model to screen candidate profiles with confidence and explainability</h3>
            <p style='margin:0; color:#475569;'>Review one applicant at a time, work through a CSV of candidates, and export a professional report for follow-up conversations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(df)
    high_income_pct = 100 * (df["income"] == ">50K").mean()
    avg_age = df["age"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Reference records", f"{total:,}")
    col2.metric("High-income share", f"{high_income_pct:.1f}%")
    col3.metric("Average age", f"{avg_age:.1f}")

    st.markdown("### Core capabilities")
    cards = [
        ("⚡", "Single prediction", "Get a fast salary-band assessment and a clear explanation for one employee."),
        ("📁", "Bulk scoring", "Upload a CSV, validate each record, and review a summary of successful predictions."),
        ("📝", "Readable reporting", "Generate PDF summaries and maintain a searchable prediction history."),
    ]
    cols = st.columns(3)
    for column, (icon, title, content) in zip(cols, cards):
        with column:
            st.markdown(f"<div class='feature-card'><div style='font-size:1.2rem'>{icon}</div><div style='font-weight:700; margin-top:0.25rem'>{title}</div><div style='color:#64748b; margin-top:0.3rem'>{content}</div></div>", unsafe_allow_html=True)

    st.markdown("### How it works")
    steps = [
        ("1", "Enter or upload employee details"),
        ("2", "Review the prediction, confidence score, and explanation"),
        ("3", "Use the generated report and history for follow-up decisions"),
    ]
    for step, text in steps:
        st.markdown(f"<div class='card'><strong>{step}.</strong> {text}</div>", unsafe_allow_html=True)

    st.markdown("### Key factors used by the model")
    importance_frame = importance.head(6).reset_index()
    importance_frame.columns = ["Feature", "Importance"]
    st.dataframe(importance_frame, use_container_width=True, hide_index=True)


def display_single_prediction(df: pd.DataFrame, encoders: dict, model, importance: pd.Series):
    st.markdown("<div class='header-title'>📊 Single Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Predict a salary band for one employee and review confidence plus explainable factors.</div>", unsafe_allow_html=True)

    with st.form("single_prediction_form"):
        left, right = st.columns(2)
        age = left.number_input("Age", min_value=18, max_value=100, value=30, help="Age is used as a general proxy for career stage.")
        workclass = left.selectbox("Workclass", sorted(df["workclass"].unique()), help="Primary employment category")
        education = left.selectbox("Education Level", list(EDUCATION_MAPPING.keys()), help="Highest completed education level")
        marital_status = left.selectbox("Marital Status", sorted(df["marital-status"].unique()), help="Relationship and household context")
        occupation = right.selectbox("Occupation", sorted(df["occupation"].unique()), help="Current role or job family")
        relationship = right.selectbox("Relationship", sorted(df["relationship"].unique()), help="Personal relationship status")
        race = right.selectbox("Race", sorted(df["race"].unique()), help="Used by the source data and kept for compatibility")
        gender = right.selectbox("Gender", sorted(df["gender"].unique()), help="Gender category from the reference data")
        capital_gain = left.number_input("Capital Gain", min_value=0, max_value=99999, value=0, step=100)
        capital_loss = right.number_input("Capital Loss", min_value=0, max_value=99999, value=0, step=100)
        hours_per_week = left.number_input("Hours per Week", min_value=1, max_value=99, value=40)
        native_country = right.selectbox("Native Country", sorted(df["native-country"].unique()))
        submit = st.form_submit_button("Predict Salary")

    if submit:
        input_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "workclass": workclass,
                    "educational-num": EDUCATION_MAPPING[education],
                    "marital-status": marital_status,
                    "occupation": occupation,
                    "relationship": relationship,
                    "race": race,
                    "gender": gender,
                    "capital-gain": capital_gain,
                    "capital-loss": capital_loss,
                    "hours-per-week": hours_per_week,
                    "native-country": native_country,
                }
            ]
        )

        input_encoded = encode_features(input_data, encoders)
        if (input_encoded == -1).any(axis=None):
            st.error("One or more categorical inputs could not be encoded. Please verify the selected values.")
            return

        with st.spinner("Generating prediction..."):
            prediction = model.predict(input_encoded)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = max(model.predict_proba(input_encoded)[0])

        factor_scores = generate_factor_summary(input_data.iloc[0], importance)
        explanation = build_prediction_explanation(prediction, factor_scores, confidence)
        recommendations = build_recommendations(prediction, confidence)

        st.markdown("---")
        st.markdown(
            f"""
            <div class='result-card'>
                <div class='result-title'>Predicted income band</div>
                <div class='result-value'>{prediction}</div>
                <div class='result-subtitle'>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success("Prediction complete")

        col_a, col_b = st.columns([1.2, 0.8])
        with col_a:
            st.markdown("### Confidence")
            st.write(f"Confidence score: **{format_probability(confidence)}**")
            if confidence is not None:
                st.progress(int(min(confidence * 100, 100)))
        with col_b:
            st.markdown("### Explanation")
            st.write(explanation)

        with st.expander("Key contributing factors", expanded=True):
            for feature, impact in factor_scores[:5]:
                st.write(f"- **{feature}**: {impact:.3f}")

        with st.expander("Recommended next steps"):
            for rec in recommendations:
                st.write(f"- {rec}")

        st.markdown("### Employee details")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

        save_history(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "age": age,
                "workclass": workclass,
                "occupation": occupation,
                "hours-per-week": hours_per_week,
                "prediction": prediction,
                "confidence": format_probability(confidence),
            }
        )

        try:
            pdf_bytes = create_pdf_report(input_data, prediction, confidence, pd.DataFrame(factor_scores, columns=["Feature", "Importance"]))
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="prediction_report.pdf", mime="application/pdf")
        except Exception as exc:
            st.warning(f"Could not generate PDF report: {exc}")


def display_batch_prediction(df: pd.DataFrame, encoders: dict, model):
    st.markdown("<div class='header-title'>📁 Batch Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Upload a CSV of employee profiles, validate it, and review a summary of the batch outcomes.</div>", unsafe_allow_html=True)

    template = pd.DataFrame(columns=["age", "workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country"])
    st.download_button("Download CSV Template", template.to_csv(index=False), "salary_batch_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to begin batch scoring.")
        return

    batch_data = pd.read_csv(uploaded_file)
    st.markdown("### Uploaded data preview")
    st.dataframe(batch_data.head(), use_container_width=True)

    validation_result = validate_batch_dataframe(batch_data, df)
    if validation_result["missing_columns"]:
        st.error(f"Missing required columns: {', '.join(validation_result['missing_columns'])}")
        return

    if st.button("Validate & Predict Batch"):
        if validation_result["invalid_rows"]:
            st.warning(f"Skipped {len(validation_result['invalid_rows'])} invalid rows. Review the issues below.")
            invalid_df = pd.DataFrame(validation_result["invalid_rows"])
            st.dataframe(invalid_df[["row_index", "reason", "details"]], use_container_width=True)

        valid_rows = validation_result["valid_rows"]
        if not valid_rows:
            st.info("No valid rows were available for prediction.")
            return

        results_df = batch_data.copy()
        results_df["Status"] = "invalid"
        results_df["Prediction"] = ""
        results_df["Confidence"] = ""
        results_df["Issue"] = ""

        for invalid_row in validation_result["invalid_rows"]:
            row_index = invalid_row["row_index"]
            results_df.at[row_index, "Status"] = "invalid"
            results_df.at[row_index, "Issue"] = "; ".join(invalid_row["details"])

        for row in valid_rows:
            row_frame = pd.DataFrame([row])
            row_frame = row_frame[["age", "workclass", "educational-num", "marital-status", "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country"]]
            encoded_row = encode_features(row_frame, encoders)
            prediction = model.predict(encoded_row)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probability = max(model.predict_proba(encoded_row)[0])

            row_index = row["__row_index__"]
            results_df.at[row_index, "Status"] = "valid"
            results_df.at[row_index, "Prediction"] = prediction
            results_df.at[row_index, "Confidence"] = format_probability(probability)
            results_df.at[row_index, "Issue"] = ""

            save_history(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "age": row.get("age"),
                    "workclass": row.get("workclass"),
                    "occupation": row.get("occupation"),
                    "hours-per-week": row.get("hours-per-week"),
                    "prediction": prediction,
                    "confidence": format_probability(probability),
                }
            )

        successful_predictions = int((results_df["Status"] == "valid").sum())
        failed_records = int((results_df["Status"] == "invalid").sum())
        high_income_count = int((results_df["Prediction"] == ">50K").sum())
        low_income_count = int((results_df["Prediction"] == "<=50K").sum())
        average_confidence = 0.0
        confidence_values = []
        for value in results_df.loc[results_df["Confidence"] != "", "Confidence"]:
            try:
                confidence_values.append(float(str(value).replace("%", "")) / 100.0)
            except ValueError:
                continue
        if confidence_values:
            average_confidence = sum(confidence_values) / len(confidence_values)

        st.success("Batch scoring complete")
        st.metric("Total records", len(results_df))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Successful predictions", successful_predictions)
        col2.metric("Failed records", failed_records)
        col3.metric("High income count", high_income_count)
        col4.metric("Low income count", low_income_count)
        col5.metric("Average confidence", f"{average_confidence * 100:.1f}%")

        st.markdown("### Batch results")
        st.dataframe(results_df, use_container_width=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv_bytes, "salary_batch_predictions.csv", "text/csv")

        summary_df = results_df[results_df["Status"] == "valid"].copy()
        if not summary_df.empty:
            summary_df["Prediction"] = summary_df["Prediction"].astype(str)
            fig_counts = px.bar(summary_df["Prediction"].value_counts().reset_index(), x="index", y="Prediction", title="Predicted income distribution")
            fig_conf = px.histogram(summary_df, x="Confidence", title="Prediction confidence")
            st.plotly_chart(fig_counts, use_container_width=True)
            st.plotly_chart(fig_conf, use_container_width=True)


def display_about_model(df: pd.DataFrame, model, importance: pd.Series):
    st.markdown("<div class='header-title'>ℹ️ About</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>A practical decision-support tool for screening candidate salary potential with explainable predictions.</div>", unsafe_allow_html=True)

    st.markdown("### Project overview")
    st.write("This application helps HR professionals and recruiters get a fast, explainable signal about likely salary band for a candidate profile. It is designed for screening and planning support, not as a final compensation decision tool.")

    st.markdown("### Dataset used")
    st.write("The model is trained on the UCI Adult dataset, with demographic, educational, occupational, and work-pattern features that are commonly used in salary classification scenarios.")
    st.write(f"Total records in the reference dataset: **{len(df):,}**")

    st.markdown("### ML algorithm")
    st.write(f"The current implementation uses a **{model.__class__.__name__}** classifier with encoded categorical features and feature-importance guidance for each prediction.")

    st.markdown("### Features considered")
    features = ["age", "workclass", "education", "marital status", "occupation", "relationship", "race", "gender", "capital gain", "capital loss", "hours per week", "native country"]
    st.write("- " + "\n- ".join(features))

    st.markdown("### Technology stack")
    st.write("Streamlit for the user experience, Plotly for visuals, pandas and scikit-learn for data handling and modeling, and FPDF for report generation.")

    st.markdown("### Limitations")
    st.write("- Predictions should be used as decision support rather than a final compensation judgment.")
    st.write("- The model is based on historical data and may reflect biases present in the source dataset.")
    st.write("- Results are best interpreted alongside interviews, role levels, and market benchmarks.")

    st.markdown("### Future enhancements")
    st.write("- Add richer role-specific salary benchmarks.")
    st.write("- Support company-specific calibration for internal hiring workflows.")
    st.write("- Add export workflows for recruiter reporting and team collaboration.")

    st.markdown("### Key drivers used in the app")
    importance_frame = importance.head(8).reset_index()
    importance_frame.columns = ["Feature", "Importance"]
    st.dataframe(importance_frame, use_container_width=True, hide_index=True)

    st.markdown("### Architecture snapshot")
    st.image("assets/architecture.png", use_container_width=True)


def display_history():
    st.markdown("<div class='header-title'>📝 Prediction History</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Review saved predictions, search for a specific record, and export the history.</div>", unsafe_allow_html=True)

    history = load_history()
    if history.empty:
        st.info("No prediction history has been saved yet.")
        return

    search_term = st.text_input("Search history")
    sort_option = st.selectbox("Sort by", ["Newest first", "Oldest first", "Prediction", "Age"])

    display_df = history.copy()
    if search_term:
        display_df = display_df[
            display_df["occupation"].astype(str).str.contains(search_term, case=False, na=False)
            | display_df["prediction"].astype(str).str.contains(search_term, case=False, na=False)
            | display_df["age"].astype(str).str.contains(search_term, case=False, na=False)
        ]

    if sort_option == "Oldest first":
        display_df = display_df.sort_values("timestamp", ascending=True)
    elif sort_option == "Prediction":
        display_df = display_df.sort_values("prediction", ascending=True)
    elif sort_option == "Age":
        display_df = display_df.sort_values("age", ascending=True)
    else:
        display_df = display_df.sort_values("timestamp", ascending=False)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history CSV", csv_bytes, "prediction_history.csv", "text/csv")

    if st.button("Clear history"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("Prediction history cleared.")
        st.rerun()


def main():
    st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")
    style_app()

    model = load_model(MODEL_PATH)
    encoders = load_encoders()
    df = load_dataset(DATA_PATH)
    importance = calculate_importance(model, encode_features(df[FEATURE_COLUMNS], encoders), df["income"])

    with st.sidebar:
        st.markdown("<div class='logo-text'>💼 Employee Salary Predictor</div>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigation", ["Home", "Single Prediction", "Batch Prediction", "History", "About"])
        st.caption("Built for recruiter-ready screening support")

    if page == "Home":
        display_home(df, importance)
    elif page == "Single Prediction":
        display_single_prediction(df, encoders, model, importance)
    elif page == "Batch Prediction":
        display_batch_prediction(df, encoders, model)
    elif page == "History":
        display_history()
    elif page == "About":
        display_about_model(df, model, importance)


if __name__ == "__main__":
    main()
