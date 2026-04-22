import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode raw transaction data."""
    df = df.copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"]       = df["trans_date_trans_time"].dt.month

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    df["geo_distance"] = _haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    drop_cols = ["trans_date_trans_time", "cc_num", "first", "last",
                 "street", "dob", "trans_num", "unix_time"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in ["merchant", "category", "gender", "city", "state", "job"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    logger.info(f"Preprocessed shape: {df.shape}")
    return df

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))
