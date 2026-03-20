"""
frontend/app.py
---------------
Streamlit frontend for the Multi-Modal Credit Scoring system.
Calls the FastAPI backend hosted on Render.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("CREDIT_API_URL", "http://localhost:8000")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
)

# ── Risk Helper ───────────────────────────────────────────────────────────────
def get_risk_category(probability: float):
    if probability < 0.3:
        return "Low Risk", "Loan application looks favourable. Recommend approval."
    elif probability < 0.6:
        return "Medium Risk", "Moderate default risk. Further review recommended."
    else:
        return "High Risk", "High probability of default. Recommend rejection or collateral."


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Multi-Modal Credit Risk Scorer")
st.markdown(
    "Predicts the probability of loan default by fusing **numerical financial data** "
    "and **categorical behavioural data** through a dual-branch neural network."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Model")

    st.markdown("**The Problem**")
    st.markdown(
        "Traditional credit scoring uses only numbers — income, loan amount. "
        "This ignores *who* the applicant is: their occupation, education, "
        "employment type. This model fuses both."
    )

    st.divider()

    st.markdown("**Architecture: Late Fusion ANN**")
    st.markdown("""
**Branch 1 — Numerical**
- Income, Credit Amount, Annuity
- Age, Employment Duration, ID Age
- External Bureau Scores (1, 2, 3)
- Loan-to-Goods Ratio
- Missingness flags for bureau scores

Processed through Dense layers with BatchNorm + Dropout.

---

**Branch 2 — Categorical**
- Occupation Type (19 categories)
- Organization Type (58 categories)
- Education Level (5 categories)
- Family Status (6 categories)
- Income Type (8 categories)
- Gender (2 categories)

Each category learned as an **Entity Embedding** — a dense vector representation that captures similarity between categories (e.g. "Accountants" and "Managers" are closer than "Accountants" and "Laborers").

---

**Fusion**
Both branches concatenated → Dense(64) → Dense(32) → Sigmoid output.

---

**Why not just one branch?**
A single branch with one-hot encoding of 58 occupation types adds 58 sparse columns. Entity embeddings compress this to 29 dense meaningful dimensions and generalise better to unseen category combinations.
    """)

    st.divider()

    st.markdown("**Training Details**")
    st.markdown("""
- Dataset: Home Credit Default Risk (Kaggle)
- 307,511 loan applications
- Class imbalance: 91% repaid / 9% default
- Handled via balanced class weights
- Validation AUC: **0.7535**
- Fairness: Demographic Parity evaluated across gender groups (DPD: 0.23)
    """)

    st.divider()

    # API health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            st.success("API Connected")
        else:
            st.warning("API reachable but model not loaded")
    except Exception:
        st.error("API not reachable.")
        st.caption(f"Trying: {API_URL}")

# ── Form ──────────────────────────────────────────────────────────────────────
st.subheader("Applicant Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Financial Details**")
    AMT_INCOME_TOTAL = st.number_input(
        "Annual Income", min_value=10000.0, max_value=10000000.0,
        value=200000.0, step=5000.0
    )
    AMT_CREDIT = st.number_input(
        "Loan Amount Requested", min_value=10000.0, max_value=5000000.0,
        value=300000.0, step=5000.0
    )
    AMT_ANNUITY = st.number_input(
        "Monthly Annuity Payment", min_value=1000.0, max_value=500000.0,
        value=15000.0, step=500.0
    )
    AMT_GOODS_PRICE = st.number_input(
        "Goods Price", min_value=10000.0, max_value=5000000.0,
        value=280000.0, step=5000.0
    )

    # Real-time Loan-to-Goods Ratio
    ltv = AMT_CREDIT / AMT_GOODS_PRICE if AMT_GOODS_PRICE > 0 else 0
    ltv_color = "normal" if ltv <= 1.0 else "inverse"
    st.metric(
        label="Loan-to-Goods Ratio (LTV)",
        value=f"{ltv:.3f}",
        delta="Within limit" if ltv <= 1.0 else "Exceeds goods value",
        delta_color=ltv_color,
        help="Computed as Loan Amount / Goods Price. Values above 1.0 mean the loan exceeds the asset value — a risk signal."
    )

with col2:
    st.markdown("**Personal Details**")
    age_years = st.slider("Applicant Age (years)", 18, 70, 35)
    DAYS_BIRTH = -(age_years * 365)

    years_employed = st.slider(
        "Years Employed (0 = currently unemployed)", 0, 40, 5
    )
    DAYS_EMPLOYED = 365243 if years_employed == 0 else -(years_employed * 365)

    years_id = st.slider("Years Since ID Was Published", 0, 20, 5)
    DAYS_ID_PUBLISH = -(years_id * 365)

    CODE_GENDER = st.selectbox("Gender", ["M", "F"])

with col3:
    st.markdown("**Background Details**")

    OCC = [
        "Accountants", "Cleaning staff", "Cooking staff", "Core staff",
        "Drivers", "HR staff", "High skill tech staff", "IT staff",
        "Laborers", "Low-skill Laborers", "Managers", "Medicine staff",
        "Private service staff", "Realty agents", "Sales staff",
        "Secretaries", "Security staff", "Waiters/barmen staff", "Unknown"
    ]
    OCCUPATION_TYPE = st.selectbox("Occupation Type", OCC, index=8)

    ORG = [
        "Agriculture", "Bank", "Business Entity Type 1",
        "Business Entity Type 2", "Business Entity Type 3",
        "Construction", "Government", "Housing", "Industry: type 1",
        "Kindergarten", "Medicine", "Military", "Mobile", "Other",
        "Police", "Postal", "Religion", "Restaurant", "School",
        "Security", "Self-employed", "Services", "Trade: type 7",
        "Transport: type 2", "University", "XNA"
    ]
    ORGANIZATION_TYPE = st.selectbox("Organization Type", ORG, index=6)

    EDU = [
        "Academic degree", "Higher education", "Incomplete higher",
        "Lower secondary", "Secondary / secondary special"
    ]
    NAME_EDUCATION_TYPE = st.selectbox("Education Level", EDU, index=1)

    FAM = [
        "Civil marriage", "Married", "Separated",
        "Single / not married", "Unknown", "Widow"
    ]
    NAME_FAMILY_STATUS = st.selectbox("Family Status", FAM, index=1)

    INC = [
        "Businessman", "Commercial associate", "Maternity leave",
        "Pensioner", "State servant", "Student", "Unemployed", "Working"
    ]
    NAME_INCOME_TYPE = st.selectbox("Income Type", INC, index=7)

st.divider()

# ── Bureau Scores ─────────────────────────────────────────────────────────────
st.subheader("External Credit Bureau Scores (Optional)")
st.caption(
    "Normalised scores (0–1) from external credit bureaus. "
    "Higher = better credit history. These are the strongest predictors in the model. "
    "Uncheck if the score is unavailable for this applicant."
)

e1, e2, e3 = st.columns(3)
with e1:
    ext1_on = st.checkbox("Bureau Score 1 available", value=True)
    EXT_SOURCE_1 = st.slider(
        "Bureau Score 1 (Normalised)", 0.0, 1.0, 0.5,
        help="External credit bureau score from source 1. Higher is better."
    ) if ext1_on else None

with e2:
    ext2_on = st.checkbox("Bureau Score 2 available", value=True)
    EXT_SOURCE_2 = st.slider(
        "Bureau Score 2 (Normalised)", 0.0, 1.0, 0.5,
        help="External credit bureau score from source 2. Higher is better."
    ) if ext2_on else None

with e3:
    ext3_on = st.checkbox("Bureau Score 3 available", value=True)
    EXT_SOURCE_3 = st.slider(
        "Bureau Score 3 (Normalised)", 0.0, 1.0, 0.5,
        help="External credit bureau score from source 3. Higher is better."
    ) if ext3_on else None

st.divider()

# ── Input Validation ──────────────────────────────────────────────────────────
validation_errors = []

if years_employed > (age_years - 16):
    validation_errors.append(
        f"Years Employed ({years_employed}) cannot exceed working age. "
        f"Applicant is {age_years} years old — maximum possible employment is {age_years - 16} years."
    )

if AMT_CREDIT > AMT_INCOME_TOTAL * 20:
    validation_errors.append(
        f"Loan amount ({AMT_CREDIT:,.0f}) is more than 20x annual income ({AMT_INCOME_TOTAL:,.0f}). "
        "This is an unrealistic application."
    )

if AMT_ANNUITY > AMT_INCOME_TOTAL / 12:
    validation_errors.append(
        f"Monthly annuity ({AMT_ANNUITY:,.0f}) exceeds monthly income ({AMT_INCOME_TOTAL/12:,.0f}). "
        "Applicant cannot afford repayments."
    )

if age_years < 18:
    validation_errors.append("Applicant must be at least 18 years old.")

if validation_errors:
    for error in validation_errors:
        st.warning(f"Input Issue: {error}")

# ── Predict Button ────────────────────────────────────────────────────────────
predict_disabled = len(validation_errors) > 0
if predict_disabled:
    st.info("Fix the input issues above before running the assessment.")

if st.button(
    "Assess Credit Risk",
    type="primary",
    use_container_width=True,
    disabled=predict_disabled
):
    payload = {
        "AMT_INCOME_TOTAL":    AMT_INCOME_TOTAL,
        "AMT_CREDIT":          AMT_CREDIT,
        "AMT_ANNUITY":         AMT_ANNUITY,
        "AMT_GOODS_PRICE":     AMT_GOODS_PRICE,
        "DAYS_BIRTH":          int(DAYS_BIRTH),
        "DAYS_EMPLOYED":       int(DAYS_EMPLOYED),
        "DAYS_ID_PUBLISH":     int(DAYS_ID_PUBLISH),
        "EXT_SOURCE_1":        EXT_SOURCE_1,
        "EXT_SOURCE_2":        EXT_SOURCE_2,
        "EXT_SOURCE_3":        EXT_SOURCE_3,
        "OCCUPATION_TYPE":     OCCUPATION_TYPE,
        "ORGANIZATION_TYPE":   ORGANIZATION_TYPE,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS":  NAME_FAMILY_STATUS,
        "NAME_INCOME_TYPE":    NAME_INCOME_TYPE,
        "CODE_GENDER":         CODE_GENDER,
    }

    with st.spinner("Running inference... (first request may take up to 30s if API is waking up)"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result         = response.json()
                prob           = result["default_probability"]
                risk           = result["risk_category"]
                recommendation = result["recommendation"]

                st.subheader("Assessment Result")
                r1, r2 = st.columns([1, 2])

                with r1:
                    if risk == "Low Risk":
                        st.success(f"### {risk}")
                    elif risk == "Medium Risk":
                        st.warning(f"### {risk}")
                    else:
                        st.error(f"### {risk}")

                    st.metric("Default Probability", f"{prob * 100:.1f}%")

                    # Input summary
                    st.markdown("**Input Summary**")
                    st.markdown(f"""
- Age: {age_years} years
- Employment: {years_employed} years
- LTV Ratio: {ltv:.3f}
- Bureau Scores: {EXT_SOURCE_1 or 'N/A':.2f} / {EXT_SOURCE_2 or 'N/A':.2f} / {EXT_SOURCE_3 or 'N/A':.2f}
                    """)

                with r2:
                    st.info(f"**Recommendation:** {recommendation}")
                    st.markdown("**Risk Gauge**")
                    st.progress(prob)
                    st.markdown("""
| Threshold | Category |
|-----------|----------|
| < 30% | Low Risk |
| 30%–60% | Medium Risk |
| > 60% | High Risk |
                    """)

            else:
                st.error(
                    f"API Error {response.status_code}: "
                    f"{response.json().get('detail', 'Unknown error')}"
                )

        except requests.exceptions.Timeout:
            st.warning(
                "The API is waking up from sleep (Render free tier cold start). "
                "Please wait 30 seconds and try again."
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with TensorFlow · FastAPI · Streamlit | "
    "Home Credit Default Risk Dataset (Kaggle) | "
    "Dual-Branch ANN + Entity Embeddings + Fairness Evaluation"
)