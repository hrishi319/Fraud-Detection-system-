import pandas as pd
from src.utils.config import TRAIN_FILE, TEST_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs."""
    logger.info("Loading raw data...")
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    logger.info(f"Train: {train.shape} | Test: {test.shape}")
    logger.info(f"Fraud rate (train): {train['is_fraud'].mean():.4%}")
    return train, test
