"""
app/schemas.py
Pydantic request/response schemas for the FastAPI endpoints.
Validation happens automatically — FastAPI rejects malformed requests
before they reach the route handler.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Input schema for /predict and /investigate endpoints.
    All fields match the preprocessed feature set from preprocessing.py.
    """
    transaction_id : str   = Field(..., description="Unique transaction identifier")
    merchant       : float = Field(..., description="Frequency-encoded merchant")
    category       : float = Field(..., description="Frequency-encoded category")
    amt            : float = Field(..., description="Transaction amount in USD")
    gender         : float = Field(..., description="Label-encoded gender")
    city           : float = Field(..., description="Frequency-encoded city")
    state          : float = Field(..., description="Frequency-encoded state")
    zip            : float = Field(..., description="ZIP code")
    city_pop       : float = Field(..., description="City population")
    job            : float = Field(..., description="Frequency-encoded job")
    hour           : float = Field(..., description="Hour of transaction (0-23)")
    day_of_week    : float = Field(..., description="Day of week (0=Mon, 6=Sun)")
    month          : float = Field(..., description="Month (1-12)")
    is_weekend     : float = Field(..., description="1 if weekend, 0 otherwise")
    is_night       : float = Field(..., description="1 if night (22:00-03:00)")
    age            : float = Field(..., description="Cardholder age in years")
    geo_distance   : float = Field(..., description="Distance from home to merchant (km)")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_001685",
                "merchant"      : 0.0012,
                "category"      : 0.0662,
                "amt"           : 24.84,
                "gender"        : 0.0,
                "city"          : 0.0016,
                "state"         : 0.045,
                "zip"           : 29307.0,
                "city_pop"      : 333497.0,
                "job"           : 0.003,
                "hour"          : 22.0,
                "day_of_week"   : 6.0,
                "month"         : 6.0,
                "is_weekend"    : 1.0,
                "is_night"      : 1.0,
                "age"           : 50.0,
                "geo_distance"  : 80.6
            }
        }


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""
    transaction_id    : str
    fraud_probability : float
    prediction        : int
    risk_level        : str
    threshold         : float


class InvestigateResponse(BaseModel):
    """Response schema for /investigate endpoint."""
    transaction_id : str
    verdict        : str
    confidence     : float
    fraud_score    : float
    risk_level     : str
    risk_signals   : list
    top_features   : list
    explanation    : str
    timestamp      : str


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status        : str
    model_loaded  : bool
    scaler_loaded : bool
    version       : str
