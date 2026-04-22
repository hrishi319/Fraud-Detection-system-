from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(model, X_val, y_val) -> dict:
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    logger.info("\n" + classification_report(y_val, y_pred, target_names=["Legit","Fraud"]))
    roc = roc_auc_score(y_val, y_prob)
    pr  = average_precision_score(y_val, y_prob)
    logger.info(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_val, y_pred)}")
    return {"roc_auc": roc, "pr_auc": pr}
