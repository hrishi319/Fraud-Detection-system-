"""
src/data/preprocessor.py
Transforms raw transaction data into model-ready features.
Matches exactly what notebooks/02_preprocessing.ipynb produced.
"""
import pandas as pd
import numpy as np
from src.utils.config import DROP_COLS, LABEL_COLS, FREQ_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _haversine(lat1, lon1, lat2, lon2) -> pd.Series:
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]       = df["trans_date_trans_time"].dt.hour
    df["day_of_week"]= df["trans_date_trans_time"].dt.dayofweek
    df["month"]      = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 3)).astype(int)
    df = df.drop(columns=["trans_date_trans_time"])
    return df


def extract_age(df: pd.DataFrame, ref_date: str = "2020-06-21") -> pd.DataFrame:
    df = df.copy()
    df["dob"] = pd.to_datetime(df["dob"])
    ref = pd.Timestamp(ref_date)
    df["age"] = (ref - df["dob"]).dt.days // 365
    df = df.drop(columns=["dob"])
    return df


def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["geo_distance"] = _haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df = df.drop(columns=["lat", "long", "merch_lat", "merch_long"])
    return df


def encode_features(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Encode categorical features.
    Fit frequency maps on train only — never on test (avoids leakage).
    Returns encoded train, encoded test, and freq_maps for inference.
    """
    train, test = train.copy(), test.copy()

    # Label encode low-cardinality
    for col in LABEL_COLS:
        train[col] = train[col].astype("category").cat.codes
        test[col]  = test[col].astype("category").cat.codes

    # Frequency encode high-cardinality
    freq_maps = {}
    for col in FREQ_COLS:
        freq_map       = train[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq_map
        train[col]     = train[col].map(freq_map)
        test[col]      = test[col].map(freq_map).fillna(0)

    return train, test, freq_maps


def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full preprocessing pipeline.
    Returns: processed train, processed test, freq_maps
    """
    logger.info("Starting preprocessing...")

    # Drop PII and identifier columns
    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    test  = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

    # Feature engineering
    for df_name, df in [("train", train), ("test", test)]:
        pass  # apply below

    train = extract_datetime_features(train)
    test  = extract_datetime_features(test)

    train = extract_age(train)
    test  = extract_age(test)

    train = add_geo_distance(train)
    test  = add_geo_distance(test)

    # Encoding
    train, test, freq_maps = encode_features(train, test)

    # Drop remaining non-numeric
    train = train.select_dtypes(include=[np.number])
    test  = test.select_dtypes(include=[np.number])

    logger.info(f"Train: {train.shape} | Test: {test.shape}")
    return train, test, freq_maps


def preprocess_single(
    transaction: dict,
    freq_maps: dict
) -> pd.DataFrame:
    """
    Preprocess a single transaction dict for inference.
    Used by the Flask API and agent layer in Phase 5/6.
    """
    df = pd.DataFrame([transaction])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                 errors="ignore")
    df = extract_datetime_features(df)
    df = extract_age(df)
    df = add_geo_distance(df)

    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    for col in FREQ_COLS:
        if col in df.columns and col in freq_maps:
            df[col] = df[col].map(freq_maps[col]).fillna(0)

    return df.select_dtypes(include=[np.number])
