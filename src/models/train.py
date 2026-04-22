import joblib
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.config import TARGET_COL, RANDOM_STATE, TEST_SIZE, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    logger.info("Applying SMOTE...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    logger.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,  MODELS_DIR / "xgboost_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    logger.info("Model and scaler saved.")
    return model, scaler, X_val, y_val
