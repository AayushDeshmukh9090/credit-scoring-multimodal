"""
app.py
------
FastAPI backend for the Multi-Modal Credit Scoring system.

Endpoints:
  GET  /health   — confirms the API and model are loaded
  POST /predict  — takes applicant data, returns default probability

Run locally:
  uvicorn api.app:app --reload --port 8000

Then open: http://localhost:8000/docs  for interactive API documentation.
"""

import sys
from pathlib import Path

# Ensure src/ is importable regardless of where uvicorn is launched from
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.preprocessor import CreditPreprocessor
from src.model_builder import prepare_model_inputs
from api.schemas import ApplicantInput, PredictionResponse, HealthResponse

# ── App Initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal Credit Scoring API",
    description="Dual-branch ANN with entity embeddings for credit default prediction.",
    version="1.0.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model and Preprocessor at Startup ───────────────────────────────────
# These are loaded once when the server starts — not on every request.

MODEL_PATH = ROOT / "models" / "best_model.keras"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
ENCODERS_PATH = ROOT / "models" / "encoder_objects.pkl"

model = None
preprocessor = None


@app.on_event("startup")
async def load_artifacts():
    global model, preprocessor
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        preprocessor = CreditPreprocessor.load(
            scaler_path=str(SCALER_PATH),
            encoders_path=str(ENCODERS_PATH),
        )
        print("[API] Model and preprocessor loaded successfully.")
    except Exception as e:
        print(f"[API] ERROR loading artifacts: {e}")


# ── Helper: Risk Categorisation ───────────────────────────────────────────────

def get_risk_category(probability: float) -> tuple[str, str]:
    """
    Convert raw probability into a human-readable risk category.

    Thresholds are intentionally conservative for a credit scoring context
    — it is better to flag more applicants as medium/high risk than to
    miss actual defaulters.
    """
    if probability < 0.3:
        return "Low Risk", "Loan application looks favourable. Recommend approval."
    elif probability < 0.6:
        return "Medium Risk", "Moderate default risk detected. Further review recommended."
    else:
        return "High Risk", "High probability of default. Recommend rejection or collateral."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantInput):
    """
    Predict credit default probability for a single applicant.

    Accepts raw applicant data (no preprocessing needed on client side).
    Returns default probability, risk category, and recommendation.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    try:
        # 1. Convert input to DataFrame (preprocessor expects a DataFrame)
        input_dict = applicant.model_dump()
        df_input = pd.DataFrame([input_dict])

        # 2. Preprocess using the fitted preprocessor (no fitting, transform only)
        df_processed = preprocessor.transform(df_input, has_target=False)

        # 3. Split into numerical and categorical
        from src.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
        X_num = df_processed[NUMERICAL_FEATURES]
        X_cat = df_processed[CATEGORICAL_FEATURES]

        # 4. Prepare inputs in the format the model expects
        model_inputs = prepare_model_inputs(X_num, X_cat)

        # 5. Run inference
        probability = float(model.predict(model_inputs, verbose=0)[0][0])

        # 6. Categorise risk
        risk_category, recommendation = get_risk_category(probability)

        return PredictionResponse(
            default_probability=round(probability, 4),
            risk_category=risk_category,
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))