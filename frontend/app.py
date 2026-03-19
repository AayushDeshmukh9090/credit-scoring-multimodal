"""
frontend/app.py
---------------
Streamlit frontend for the Multi-Modal Credit Scoring system.

Run locally:
  streamlit run frontend/app.py

Make sure the FastAPI backend is running first:
  uvicorn api.app:app --reload --port 8000
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Multi-Modal Credit Risk Scorer")
st.markdown(
    "A dual-branch neural network combining numerical and categorical features "
    "to predict the probability of loan default."
)
st.divider()

# ── Sidebar: Model Info ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    **Architecture:** Dual-Branch ANN with Entity Embeddings
    
    **Dataset:** Home Credit Default Risk (Kaggle)
    
    **Validation AUC:** 0.7535
    
    **Fairness Check:** Demographic Parity evaluated on gender groups
    
    ---
    
    **Branch 1 — Numerical**  
    Income, credit amount, external scores, employment history
    
    **Branch 2 — Categorical**  
    Occupation, education, family status → Entity Embeddings
    
    **Fusion**  
    Both branches concatenated before final decision layers
    """)

    st.divider()

    # Health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            st.success("API Connected")
        else:
            st.warning("API reachable but model not loaded")
    except Exception:
        st.error("API not reachable. Start the backend first.")

# ── Input Form ────────────────────────────────────────────────────────────────

st.subheader("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Financial Details**")

    AMT_INCOME_TOTAL = st.number_input(
        "Annual Income (₹ or currency units)",
        min_value=10000.0,
        max_value=10000000.0,
        value=270000.0,
        step=5000.0,
    )
    AMT_CREDIT = st.number_input(
        "Loan Amount Requested",
        min_value=10000.0,
        max_value=5000000.0,
        value=300000.0,
        step=5000.0,
    )
    AMT_ANNUITY = st.number_input(
        "Monthly Annuity Payment",
        min_value=1000.0,
        max_value=500000.0,
        value=15000.0,
        step=500.0,
    )
    AMT_GOODS_PRICE = st.number_input(
        "Goods Price",
        min_value=10000.0,
        max_value=5000000.0,
        value=351000.0,
        step=5000.0,
    )

with col2:
    st.markdown("**Personal Details**")

    age_years = st.slider("Age (Years)", min_value=18, max_value=70, value=35)
    DAYS_BIRTH = -(age_years * 365)

    years_employed = st.slider("Years Employed", min_value=0, max_value=40, value=8)
    
    if years_employed == 0:
        DAYS_EMPLOYED = 365243  # sentinel for unemployed
    else:
        DAYS_EMPLOYED = -(years_employed * 365)

    years_id = st.slider(
        "Years Since ID Published",
        min_value=0, max_value=20, value=5
    )
    DAYS_ID_PUBLISH = -(years_id * 365)

    CODE_GENDER = st.selectbox("Gender", options=["M", "F"])

with col3:
    st.markdown("**Background Details**")

    OCCUPATION_TYPE = st.selectbox("Occupation Type", options=[
        "Laborers", "Core staff", "Accountants", "Managers",
        "Drivers", "Sales staff", "Cleaning staff", "Cooking staff",
        "Private service staff", "Medicine staff", "Security staff",
        "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers",
        "Realty agents", "Secretaries", "IT staff", "HR staff", "Unknown"
    ])

    ORGANIZATION_TYPE = st.selectbox("Organization Type", options=[
        "Business Entity Type 3", "School", "Government", "Religion",
        "Other", "Medicine", "Business Entity Type 2", "Self-employed",
        "Transport: type 2", "Construction", "Housing",
        "Kindergarten", "Trade: type 7", "Industry: type 11",
        "Military", "Services", "Security Ministries", "Transport: type 4",
        "Industry: type 1", "Emergency", "Security", "Trade: type 2",
        "University", "Transport: type 3", "Police", "Business Entity Type 1",
        "Postal", "Industry: type 4", "Agriculture", "Restaurant",
        "Culture", "Hotel", "Industry: type 7", "Trade: type 3",
        "Industry: type 3", "Bank", "Industry: type 9", "Insurance",
        "Trade: type 6", "Industry: type 2", "Transport: type 1",
        "Industry: type 12", "Mobile", "Trade: type 1", "Industry: type 5",
        "Industry: type 10", "Legal Services", "Advertising",
        "Trade: type 5", "Cleaning", "Industry: type 13",
        "Trade: type 4", "Telecom", "Industry: type 8",
        "Realtor", "Industry: type 6", "Insurance", "XNA"
    ])

    NAME_EDUCATION_TYPE = st.selectbox("Education Level", options=[
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree"
    ])

    NAME_FAMILY_STATUS = st.selectbox("Family Status", options=[
        "Single / not married",
        "Married",
        "Civil marriage",
        "Widow",
        "Separated",
        "Unknown"
    ])

    NAME_INCOME_TYPE = st.selectbox("Income Type", options=[
        "Working", "State servant", "Commercial associate",
        "Pensioner", "Unemployed", "Student",
        "Businessman", "Maternity leave"
    ])

st.divider()

# ── External Credit Scores (Optional) ────────────────────────────────────────

st.subheader("External Credit Scores (Optional)")
st.caption("These are external bureau scores between 0 and 1. Leave blank if unavailable.")

ecol1, ecol2, ecol3 = st.columns(3)

with ecol1:
    ext1_available = st.checkbox("EXT_SOURCE_1 available", value=True)
    EXT_SOURCE_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.65) if ext1_available else None

with ecol2:
    ext2_available = st.checkbox("EXT_SOURCE_2 available", value=True)
    EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.70) if ext2_available else None

with ecol3:
    ext3_available = st.checkbox("EXT_SOURCE_3 available", value=True)
    EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.65) if ext3_available else None

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────

predict_clicked = st.button("Assess Credit Risk", type="primary", use_container_width=True)

if predict_clicked:
    payload = {
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
        "DAYS_BIRTH": int(DAYS_BIRTH),
        "DAYS_EMPLOYED": int(DAYS_EMPLOYED),
        "DAYS_ID_PUBLISH": int(DAYS_ID_PUBLISH),
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "OCCUPATION_TYPE": OCCUPATION_TYPE,
        "ORGANIZATION_TYPE": ORGANIZATION_TYPE,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
        "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
        "CODE_GENDER": CODE_GENDER,
    }

    with st.spinner("Running inference..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                prob = result["default_probability"]
                risk = result["risk_category"]
                recommendation = result["recommendation"]

                st.subheader("Assessment Result")

                # Colour-coded result
                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    if risk == "Low Risk":
                        st.success(f"### {risk}")
                    elif risk == "Medium Risk":
                        st.warning(f"### {risk}")
                    else:
                        st.error(f"### {risk}")

                    st.metric(
                        label="Default Probability",
                        value=f"{prob * 100:.1f}%"
                    )

                with res_col2:
                    st.info(f"**Recommendation:** {recommendation}")

                    # Probability bar
                    st.markdown("**Risk Gauge**")
                    st.progress(prob)

                    st.markdown(f"""
                    | Threshold | Category |
                    |-----------|----------|
                    | < 30%     | Low Risk |
                    | 30% – 60% | Medium Risk |
                    | > 60%     | High Risk |
                    """)

            else:
                st.error(f"API Error {response.status_code}: {response.json().get('detail')}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the backend is running: `uvicorn api.app:app --reload`")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with TensorFlow, FastAPI, and Streamlit | "
    "Home Credit Default Risk Dataset | "
    "Dual-Branch ANN + Entity Embeddings"
)