"""
frontend/app.py
---------------
Streamlit frontend for the Multi-Modal Credit Scoring system.
Calls the FastAPI backend hosted on Render.

Set the API URL via environment variable:
  CREDIT_API_URL = https://your-app.onrender.com

Locally: set CREDIT_API_URL=http://localhost:8000 in your terminal
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# API URL — reads from environment variable, falls back to localhost
API_URL = os.environ.get("CREDIT_API_URL", "http://localhost:8000")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
)

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
    "A dual-branch neural network combining **numerical** and **categorical** features "
    "to predict the probability of loan default."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    **Architecture**  
    Dual-Branch ANN + Entity Embeddings

    **Dataset**  
    Home Credit Default Risk (Kaggle)  
    307,511 loan applications

    **Validation AUC:** 0.7535

    ---

    **Branch 1 — Numerical**  
    Income, credit, annuity, external scores, employment

    **Branch 2 — Categorical**  
    Occupation, education, family status  
    → Learned via Entity Embeddings

    **Fairness Check**  
    Demographic Parity Difference: 0.23  
    Evaluated across gender groups
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

    st.divider()
    st.markdown("**Quick Scenarios**")
    col_a, col_b, col_c = st.columns(3)
    low_clicked    = col_a.button("Low")
    medium_clicked = col_b.button("Med")
    high_clicked   = col_c.button("High")

# ── Scenario Presets ──────────────────────────────────────────────────────────
SCENARIOS = {
    "low": dict(
        income=270000.0, credit=250000.0, annuity=12000.0, goods=240000.0,
        age=38, employed=10, id_years=8,
        ext1=0.72, ext2=0.75, ext3=0.70,
        occupation="Accountants", org="Government",
        education="Higher education", family="Married",
        income_type="State servant", gender="F",
    ),
    "medium": dict(
        income=135000.0, credit=400000.0, annuity=22000.0, goods=380000.0,
        age=30, employed=3, id_years=5,
        ext1=0.45, ext2=0.50, ext3=0.42,
        occupation="Sales staff", org="Business Entity Type 3",
        education="Secondary / secondary special", family="Civil marriage",
        income_type="Working", gender="M",
    ),
    "high": dict(
        income=67500.0, credit=500000.0, annuity=28000.0, goods=480000.0,
        age=25, employed=0, id_years=2,
        ext1=0.10, ext2=0.15, ext3=0.12,
        occupation="Laborers", org="Business Entity Type 3",
        education="Lower secondary", family="Single / not married",
        income_type="Unemployed", gender="M",
    ),
}

if low_clicked:
    st.session_state["scenario"] = "low"
elif medium_clicked:
    st.session_state["scenario"] = "medium"
elif high_clicked:
    st.session_state["scenario"] = "high"

s = SCENARIOS[st.session_state.get("scenario", "low")]

# ── Form ──────────────────────────────────────────────────────────────────────
st.subheader("Applicant Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Financial**")
    AMT_INCOME_TOTAL = st.number_input("Annual Income", 10000.0, 10000000.0, s["income"], 5000.0)
    AMT_CREDIT       = st.number_input("Loan Amount",   10000.0, 5000000.0,  s["credit"], 5000.0)
    AMT_ANNUITY      = st.number_input("Monthly Annuity", 1000.0, 500000.0,  s["annuity"], 500.0)
    AMT_GOODS_PRICE  = st.number_input("Goods Price",   10000.0, 5000000.0,  s["goods"],  5000.0)

with col2:
    st.markdown("**Personal**")
    age_years       = st.slider("Age (years)", 18, 70, s["age"])
    DAYS_BIRTH      = -(age_years * 365)

    years_employed  = st.slider("Years Employed (0 = unemployed)", 0, 40, s["employed"])
    DAYS_EMPLOYED   = 365243 if years_employed == 0 else -(years_employed * 365)

    years_id        = st.slider("Years Since ID Published", 0, 20, s["id_years"])
    DAYS_ID_PUBLISH = -(years_id * 365)

    CODE_GENDER = st.selectbox("Gender", ["M", "F"],
                               index=0 if s["gender"] == "M" else 1)

with col3:
    st.markdown("**Background**")

    OCC = ["Accountants","Cleaning staff","Cooking staff","Core staff","Drivers",
           "HR staff","High skill tech staff","IT staff","Laborers",
           "Low-skill Laborers","Managers","Medicine staff","Private service staff",
           "Realty agents","Sales staff","Secretaries","Security staff",
           "Waiters/barmen staff","Unknown"]
    OCCUPATION_TYPE = st.selectbox("Occupation", OCC,
                                   index=OCC.index(s["occupation"]) if s["occupation"] in OCC else 0)

    ORG = ["Agriculture","Bank","Business Entity Type 1","Business Entity Type 2",
           "Business Entity Type 3","Construction","Government","Housing",
           "Industry: type 1","Kindergarten","Medicine","Military","Mobile",
           "Other","Police","Postal","Religion","Restaurant","School",
           "Security","Self-employed","Services","Trade: type 7",
           "Transport: type 2","University","XNA"]
    ORGANIZATION_TYPE = st.selectbox("Organization Type", ORG,
                                     index=ORG.index(s["org"]) if s["org"] in ORG else 0)

    EDU = ["Academic degree","Higher education","Incomplete higher",
           "Lower secondary","Secondary / secondary special"]
    NAME_EDUCATION_TYPE = st.selectbox("Education", EDU,
                                       index=EDU.index(s["education"]) if s["education"] in EDU else 0)

    FAM = ["Civil marriage","Married","Separated","Single / not married","Unknown","Widow"]
    NAME_FAMILY_STATUS = st.selectbox("Family Status", FAM,
                                      index=FAM.index(s["family"]) if s["family"] in FAM else 0)

    INC = ["Businessman","Commercial associate","Maternity leave","Pensioner",
           "State servant","Student","Unemployed","Working"]
    NAME_INCOME_TYPE = st.selectbox("Income Type", INC,
                                    index=INC.index(s["income_type"]) if s["income_type"] in INC else 0)

st.divider()

# ── External Credit Scores ────────────────────────────────────────────────────
st.subheader("External Credit Bureau Scores (Optional)")
st.caption("Scores between 0–1. These are the strongest predictors in the model.")

e1, e2, e3 = st.columns(3)
with e1:
    ext1_on      = st.checkbox("EXT_SOURCE_1 available", value=True)
    EXT_SOURCE_1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, s["ext1"]) if ext1_on else None
with e2:
    ext2_on      = st.checkbox("EXT_SOURCE_2 available", value=True)
    EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, s["ext2"]) if ext2_on else None
with e3:
    ext3_on      = st.checkbox("EXT_SOURCE_3 available", value=True)
    EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, s["ext3"]) if ext3_on else None

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Assess Credit Risk", type="primary", use_container_width=True):

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

    with st.spinner("Running inference..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=60  # 60s timeout accounts for Render cold start
            )

            if response.status_code == 200:
                result      = response.json()
                prob        = result["default_probability"]
                risk        = result["risk_category"]
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
                st.error(f"API Error {response.status_code}: {response.json().get('detail')}")

        except requests.exceptions.Timeout:
            st.warning(
                "The API is waking up from sleep (Render free tier). "
                "Please wait 30 seconds and try again."
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with TensorFlow · FastAPI · Streamlit | "
    "Home Credit Default Risk Dataset | "
    "Dual-Branch ANN + Entity Embeddings + Fairness Evaluation"
)