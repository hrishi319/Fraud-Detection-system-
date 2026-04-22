import joblib
import pandas as pd
import numpy as np
from src.utils.config import MODELS_DIR

def load_artifacts():
    model  = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    model, scaler = load_artifacts()
    X = scaler.transform(df)
    return model.predict_proba(X)[:, 1]
