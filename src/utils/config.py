"""
src/utils/config.py
Central configuration — all constants live here.
Change once, propagates everywhere.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR        = Path(__file__).resolve().parents[2]
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models"
REPORTS_DIR     = ROOT_DIR / "reports"

TRAIN_FILE      = DATA_RAW / "fraudTrain.csv"
TEST_FILE       = DATA_RAW / "fraudTest.csv"
TRAIN_PROCESSED = DATA_PROCESSED / "train_preprocessed.csv"
TEST_PROCESSED  = DATA_PROCESSED / "test_preprocessed.csv"

# Model constants
RANDOM_STATE = 42
TEST_SIZE    = 0.2
TARGET_COL   = "is_fraud"

# Preprocessing
DROP_COLS  = ["cc_num", "first", "last", "street", "trans_num", "unix_time"]
LABEL_COLS = ["gender"]
FREQ_COLS  = ["merchant", "category", "city", "state", "job"]

# Monitoring
MONITOR_FEATURES = ["amt", "hour", "geo_distance"]
PSI_NO_DRIFT     = 0.10
PSI_MODERATE     = 0.20
PRAUC_ALERT      = 0.80

# API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
