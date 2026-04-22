import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
REPORTS_DIR    = ROOT_DIR / "reports"

TRAIN_FILE = DATA_RAW / "fraudTrain.csv"
TEST_FILE  = DATA_RAW / "fraudTest.csv"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
TARGET_COL   = "is_fraud"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
