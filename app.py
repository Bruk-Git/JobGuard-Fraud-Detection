import streamlit as st
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from scipy.sparse import issparse


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="JobGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# LOAD FINAL TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "final_decision_tree_model.joblib"
)


@st.cache_resource
def load_model():

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    threshold = float(saved_model["threshold"])

    return model, threshold


try:

    model, threshold = load_model()

except Exception as error:

    st.error(f"Could not load the final trained model: {error}")
    st.stop()


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("🛡️ JobGuard")

st.subheader("Fraudulent Online Job Advertisement Detection")

st.write(
    "Analyze an online job advertisement using a machine learning "
    "model trained with a dataset of legitimate and fraudulent job postings."
)

st.info(
    "This tool provides an automated risk assessment. "
    "Its predictions are not proof that an advertisement is fraudulent."
)

st.divider()


# --------------------------------------------------
# JOB ADVERTISEMENT FORM
# --------------------------------------------------

st.header("Enter Job Advertisement Details")

with st.form("job_analysis_form"):

    col1, col2 = st.columns(2)

    with col1:

        title = st.text_input(
            "Job Title *",
            placeholder="e.g., Software Developer"
        )

        company_profile = st.text_area(
            "Company Profile",
            placeholder="Describe the company..."
        )

        description = st.text_area(
            "Job Description *",
            placeholder="Describe the responsibilities and duties..."
        )

        requirements = st.text_area(
            "Job Requirements",
            placeholder="List the required skills and qualifications..."
        )

        benefits = st.text_area(
            "Benefits",
            placeholder="Describe salary benefits, insurance, etc."
        )

    with col2:

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Unknown",
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Other"
            ]
        )

        required_experience = st.selectbox(
            "Required Experience",
            [
                "Unknown",
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive"
            ]
        )

        required_education = st.selectbox(
            "Required Education",
            [
                "Unknown",
                "High School or equivalent",
                "Bachelor's Degree",
                "Master's Degree",
                "Doctorate",
                "Vocational",
                "Certification",
                "Unspecified"
            ]
        )

        salary_range = st.text_input(
            "Salary Range",
            placeholder="e.g., 50000-70000"
        )

        industry = st.text_input(
            "Industry",
            placeholder="e.g., Information Technology"
        )

        function = st.text_input(
            "Job Function",
            placeholder="e.g., Engineering"
        )

        telecommuting = st.checkbox(
            "Remote/telecommuting position"
        )

        has_company_logo = st.checkbox(
            "Company logo is available"
        )

        has_questions = st.checkbox(
            "Screening questions are included"
        )

    submitted = st.form_submit_button(
        "🔍 Analyze Job Advertisement",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION AND FEATURE ENGINEERING
# --------------------------------------------------

if submitted:

    if not title.strip() or not description.strip():

        st.warning(
            "Please provide at least a job title and job description."
        )

    else:

        # ------------------------------------------
        # PREPARE INPUT DATA
        # ------------------------------------------

        input_data = pd.DataFrame([{

            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,

            "salary_range": salary_range,

            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,

            "industry": industry,
            "function": function,

            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions)

        }])

        try:

            # --------------------------------------
            # FEATURE ENGINEERING
            # --------------------------------------

            text_columns = [
                "title",
                "company_profile",
                "description",
                "requirements",
                "benefits"
            ]

            for column in text_columns:

                input_data[column] = (
                    input_data[column]
                    .fillna("")
                    .astype(str)
                )

            # Missing information indicators

            input_data["company_profile_missing"] = (
                input_data["company_profile"].str.strip() == ""
            ).astype(int)

            input_data["requirements_missing"] = (
                input_data["requirements"].str.strip() == ""
            ).astype(int)

            input_data["benefits_missing"] = (
                input_data["benefits"].str.strip() == ""
            ).astype(int)

            # Text lengths

            input_data["description_length"] = (
                input_data["description"].str.len()
            )

            input_data["requirements_length"] = (
                input_data["requirements"].str.len()
            )

            input_data["benefits_length"] = (
                input_data["benefits"].str.len()
            )

            input_data["total_text_length"] = (
                input_data["title"].str.len()
                + input_data["company_profile"].str.len()
                + input_data["description"].str.len()
                + input_data["requirements"].str.len()
                + input_data["benefits"].str.len()
            )

            # Salary information

            input_data["has_salary_information"] = (
                input_data["salary_range"].notna()
                & (
                    input_data["salary_range"]
                    .astype(str)
                    .str.strip() != ""
                )
            ).astype(int)

            # Missing fields count

            missing_columns = [
                "company_profile",
                "description",
                "requirements",
                "benefits"
            ]

            input_data["missing_fields_count"] = sum(
                input_data[column].str.strip() == ""
                for column in missing_columns
            )

            # --------------------------------------
            # MAKE PREDICTION
            # --------------------------------------

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            legitimate_index = np.where(classes == 0)[0][0]
            fraudulent_index = np.where(classes == 1)[0][0]

            legitimate_probability = (
                probabilities[legitimate_index] * 100
            )

            fraudulent_probability = (
                probabilities[fraudulent_index] * 100
            )

            prediction = int(
                probabilities[fraudulent_index] >= threshold
            )

            # --------------------------------------
            # EXTRACT DECISION PATH
            # --------------------------------------

            preprocessor = model.named_steps["preprocessor"]

            tree_classifier = model.named_steps["classifier"]

            transformed_input = preprocessor.transform(input_data)

            decision_path = tree_classifier.decision_path(
                transformed_input
            )

            tree = tree_classifier.tree_

            feature_names = preprocessor.get_feature_names_out()

            # Handle sparse and dense matrices

            if issparse(transformed_input):

                feature_values = transformed_input.toarray()[0]

            else:

                feature_values = np.asarray(transformed_input)[0]

            node_indices = decision_path.indices[
                decision_path.indptr[0]:
                decision_path.indptr[1]
            ]

            explanation_rules = []

            for node_id in node_indices:

                feature_index = tree.feature[node_id]

                # Ignore leaf nodes

                if feature_index == -2:
                    continue

                feature_name = feature_names[feature_index]

                feature_value = feature_values[feature_index]

                split_threshold = tree.threshold[node_id]

                if feature_value <= split_threshold:

                    explanation_rules.append(
                        f"{feature_name}: {feature_value:.4f} "
                        f"≤ {split_threshold:.4f}"
                    )

                else:

                    explanation_rules.append(
                        f"{feature_name}: {feature_value:.4f} "
                        f"> {split_threshold:.4f}"
                    )

            # --------------------------------------
            # GET LEAF INFORMATION
            # --------------------------------------

            leaf_id = tree_classifier.apply(
                transformed_input
            )[0]

            leaf_values = tree.value[leaf_id][0]

            leaf_total = leaf_values.sum()

            if leaf_total > 0:

                leaf_proportions = leaf_values / leaf_total

            else:

                leaf_proportions = np.zeros_like(leaf_values)

            leaf_samples = tree.n_node_samples[leaf_id]

            leaf_legitimate_index = np.where(classes == 0)[0][0]
            leaf_fraudulent_index = np.where(classes == 1)[0][0]

            legitimate_proportion = (
                leaf_proportions[leaf_legitimate_index] * 100
            )

            fraudulent_proportion = (
                leaf_proportions[leaf_fraudulent_index] * 100
            )

            # --------------------------------------
            # DISPLAY MODEL DIAGNOSTICS
            # --------------------------------------

            st.divider()

            st.header("Model Diagnostic Information")

            st.write(
                "**Model:** Tuned Decision Tree Classifier"
            )

            st.write(
                f"**Classification threshold:** "
                f"{threshold * 100:.0f}%"
            )

            st.write(
                f"**Legitimate-class probability:** "
                f"{legitimate_probability:.2f}%"
            )

            st.write(
                f"**Fraudulent-class probability:** "
                f"{fraudulent_probability:.2f}%"
            )

            st.progress(
                float(
                    min(
                        max(fraudulent_probability / 100, 0.0),
                        1.0
                    )
                )
            )

            # --------------------------------------
            # DISPLAY ANALYSIS RESULT
            # --------------------------------------

            st.divider()

            st.header("Analysis Result")

            if prediction == 1:

                st.error(
                    "⚠️ Potentially Fraudulent Job Advertisement"
                )

                st.write(
                    "The model identified patterns associated with "
                    "fraudulent advertisements. Verify the employer, "
                    "official website, contact details, and any requests "
                    "for money before proceeding."
                )

            else:

                st.success(
                    "✅ Classified as Likely Legitimate"
                )

                st.write(
                    "The model classified this advertisement as legitimate. "
                    "However, this does not guarantee that the job is safe "
                    "or that the employer is genuine."
                )

            # --------------------------------------
            # DECISION TREE EXPLANATION
            # --------------------------------------

            st.divider()

            st.header("Why Did JobGuard Make This Prediction?")

            st.write(
                "The following conditions show the decision-tree "
                "branches followed when analyzing this advertisement."
            )

            with st.expander("View Decision Tree Explanation"):

                st.write(
                    f"**Final fraud probability:** "
                    f"{fraudulent_probability:.2f}%"
                )

                st.write(
                    f"**Classification threshold:** "
                    f"{threshold * 100:.0f}%"
                )

                st.write("**Decision rules followed:**")

                for index, rule in enumerate(
                    explanation_rules,
                    start=1
                ):

                    st.write(f"{index}. {rule}")

                st.divider()

                st.write("**Final Decision Tree Leaf Information**")

                st.write(f"Leaf node: {leaf_id}")

                st.write(
                    f"Training samples reaching this leaf: "
                    f"{leaf_samples}"
                )

                st.write(
                    f"Legitimate class proportion: "
                    f"{legitimate_proportion:.2f}%"
                )

                st.write(
                    f"Fraudulent class proportion: "
                    f"{fraudulent_proportion:.2f}%"
                )

                st.caption(
                    "These proportions describe the training samples "
                    "reaching this leaf. They are not guarantees about "
                    "the real-world legitimacy of the advertisement."
                )

            st.caption(
                "JobGuard is an experimental academic project. "
                "Always independently verify job advertisements."
            )

        except Exception as error:

            st.error(
                f"An error occurred while analyzing the advertisement: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "JobGuard | MSc Computer Science Machine Learning Project"
)