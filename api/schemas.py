"""
schemas.py
----------
Defines the data shapes for API requests and responses using Pydantic.

Why this exists separately:
  Pydantic validates and documents every field automatically.
  FastAPI uses these schemas to generate the /docs page.
  The frontend and any API consumer knows exactly what to send.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ApplicantInput(BaseModel):
    """
    Raw input from the frontend form.
    These are the ORIGINAL values before any preprocessing.
    The API handles all transformation internally.
    """

    # Numerical fields
    AMT_INCOME_TOTAL: float = Field(
        ..., gt=0, description="Annual income in currency units", example=202500.0
    )
    AMT_CREDIT: float = Field(
        ..., gt=0, description="Credit amount of the loan", example=406597.5
    )
    AMT_ANNUITY: float = Field(
        ..., gt=0, description="Loan annuity (monthly payment)", example=24700.5
    )
    AMT_GOODS_PRICE: float = Field(
        ..., gt=0, description="Price of goods for which the loan is given", example=351000.0
    )
    DAYS_BIRTH: int = Field(
        ..., lt=0, description="Age in days (negative value, e.g. -9461)", example=-9461
    )
    DAYS_EMPLOYED: int = Field(
        ..., description="Days employed (negative=employed, 365243=unemployed)", example=-637
    )
    DAYS_ID_PUBLISH: int = Field(
        ..., lt=0, description="Days since ID was published (negative)", example=-2120
    )
    EXT_SOURCE_1: float = Field(
        None, ge=0, le=1, description="External credit score 1 (optional)", example=0.083
    )
    EXT_SOURCE_2: float = Field(
        None, ge=0, le=1, description="External credit score 2 (optional)", example=0.263
    )
    EXT_SOURCE_3: float = Field(
        None, ge=0, le=1, description="External credit score 3 (optional)", example=0.139
    )

    # Categorical fields (raw string values, not encoded)
    OCCUPATION_TYPE: str = Field(
        ..., description="Applicant's occupation", example="Laborers"
    )
    ORGANIZATION_TYPE: str = Field(
        ..., description="Type of organization applicant works for", example="Business Entity Type 3"
    )
    NAME_EDUCATION_TYPE: str = Field(
        ..., description="Highest education level", example="Secondary / secondary special"
    )
    NAME_FAMILY_STATUS: str = Field(
        ..., description="Family status", example="Single / not married"
    )
    NAME_INCOME_TYPE: str = Field(
        ..., description="Income type", example="Working"
    )
    CODE_GENDER: Literal["M", "F"] = Field(
        ..., description="Gender", example="M"
    )


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    default_probability: float = Field(
        ..., description="Probability of loan default (0 to 1)"
    )
    risk_category: str = Field(
        ..., description="Risk bucket: Low / Medium / High"
    )
    recommendation: str = Field(
        ..., description="Human-readable recommendation"
    )


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str
    model_loaded: bool