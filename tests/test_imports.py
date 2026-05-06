"""
Smoke tests — verify all src and app modules import without errors.
These run on every push via GitHub Actions.
"""

def test_import_config():
    from src.utils.config import TARGET_COL, RANDOM_STATE
    assert TARGET_COL == "is_fraud"
    assert RANDOM_STATE == 42

def test_import_logger():
    from src.utils.logger import get_logger
    logger = get_logger("test")
    assert logger is not None

def test_import_loader():
    from src.data.loader import load_raw_data
    assert callable(load_raw_data)

def test_import_preprocessor():
    from src.data.preprocessor import preprocess
    assert callable(preprocess)

def test_import_train():
    from src.models.train import train_model
    assert callable(train_model)

def test_import_evaluate():
    from src.models.evaluate import evaluate
    assert callable(evaluate)

def test_import_predict():
    from src.models.predict import predict_proba
    assert callable(predict_proba)

def test_import_state():
    from src.agents.state import FraudInvestigationState
    assert FraudInvestigationState is not None

def test_import_schemas():
    from app.schemas import TransactionRequest, PredictResponse
    assert TransactionRequest is not None
    assert PredictResponse is not None
