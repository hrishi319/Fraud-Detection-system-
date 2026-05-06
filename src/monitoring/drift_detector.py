"""
src/monitoring/drift_detector.py

Custom Fraud Drift Detector — production-grade monitoring script.
Designed specifically for imbalanced fraud detection datasets.

Statistical test selection rationale:

  PSI (Population Stability Index)
    → Banking industry standard for distribution monitoring
    → Captures full distribution shift including tails
    → Interpretable thresholds: <0.10 stable, 0.10-0.20 moderate, >0.20 significant
    → Used in production by major banks for credit risk monitoring

  Mann-Whitney U (fraud-only comparison)
    → Non-parametric, rank-based, distribution-free
    → Works reliably on small fraud samples
    → Detects tail shifts without mean assumption (unlike Cohen's d)
    → Directly answers: has the fraud distribution shifted?

  PR-AUC Tracking (model performance)
    → Primary metric for imbalanced fraud detection
    → Direct measure of model's ability to find fraud
    → Triggers retraining alert when below threshold

Why Anderson-Darling was removed:
    → Produces negative statistics on nearly-identical large datasets
    → scipy's significance_level output is misleading (not a p-value)
    → Unreliable when drift signal is <1% of full dataset
    → Replaced by Mann-Whitney which is more appropriate for fraud-only comparison

Usage (scheduled via cron or Airflow):
  python -m src.monitoring.drift_detector \\
      --reference data/processed/train_preprocessed.csv \\
      --current   data/processed/production_batch.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score
)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET           = 'is_fraud'
MONITOR_FEATURES = ['amt', 'hour', 'geo_distance']

# PSI thresholds — banking industry standard
PSI_NO_DRIFT  = 0.10
PSI_MODERATE  = 0.20

# Mann-Whitney significance threshold
MW_THRESHOLD  = 0.05

# PR-AUC alert threshold
PRAUC_ALERT   = 0.80


# ── PSI ────────────────────────────────────────────────────────────────────────

def compute_psi(ref: pd.Series, curr: pd.Series, bins: int = 10) -> dict:
    """
    Population Stability Index — banking industry standard.

    How it works:
      1. Bin reference distribution into equal-frequency buckets
      2. Calculate % of reference falling in each bin
      3. Calculate % of current falling in each bin
      4. PSI = sum((curr% - ref%) * ln(curr% / ref%))

    Thresholds:
      PSI < 0.10  → Stable, no action needed
      PSI 0.10-0.20 → Moderate drift, investigate
      PSI > 0.20  → Significant drift, action required
    """
    breakpoints = np.nanpercentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    def bin_pct(series, breaks):
        counts = np.histogram(series, bins=breaks)[0]
        pct    = counts / len(series)
        return np.where(pct == 0, 1e-6, pct)

    ref_pct  = bin_pct(ref.dropna(),  breakpoints)
    curr_pct = bin_pct(curr.dropna(), breakpoints)

    min_len  = min(len(ref_pct), len(curr_pct))
    ref_pct  = ref_pct[:min_len]
    curr_pct = curr_pct[:min_len]

    psi_value = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))

    if psi_value < PSI_NO_DRIFT:
        level = 'STABLE'
    elif psi_value < PSI_MODERATE:
        level = 'MODERATE DRIFT'
    else:
        level = 'SIGNIFICANT DRIFT'

    return {
        'test'   : 'PSI',
        'value'  : round(psi_value, 4),
        'level'  : level,
        'drifted': psi_value >= PSI_NO_DRIFT
    }


# ── Mann-Whitney U ─────────────────────────────────────────────────────────────

def mann_whitney_test(ref: pd.Series, curr: pd.Series) -> dict:
    """
    Mann-Whitney U — for fraud-only comparison.

    Why chosen over Cohen's d:
      Cohen's d measures mean shift — fraud manifests in tails, not means.
      Mann-Whitney is rank-based — sensitive to shifts anywhere in the
      distribution including tails. No normality or mean assumption.
      Works on unequal sample sizes (7506 ref fraud vs ~54 curr fraud).

    Interpretation:
      p < 0.05 → distributions are statistically different → drift detected
    """
    ref_clean  = ref.dropna().values
    curr_clean = curr.dropna().values

    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return {
            'test'   : 'Mann-Whitney U',
            'drifted': None,
            'note'   : f'Insufficient samples (ref={len(ref_clean)}, curr={len(curr_clean)})'
        }

    statistic, p_value = stats.mannwhitneyu(
        ref_clean, curr_clean, alternative='two-sided'
    )

    return {
        'test'     : 'Mann-Whitney U',
        'statistic': round(float(statistic), 4),
        'p_value'  : round(float(p_value), 4),
        'drifted'  : p_value < MW_THRESHOLD
    }


# ── Core Detector ──────────────────────────────────────────────────────────────

class FraudDriftDetector:
    """
    Three-layer drift detector for imbalanced fraud data.

    Layer 1 — PSI (full dataset)
        Population-level stability monitoring.
        Banking industry standard. Primary drift signal.
        Limitation: may not detect fraud-only drift on heavily imbalanced data.

    Layer 2 — Mann-Whitney U (fraud-only, full reference fraud history)
        Uses ALL historical fraud rows as reference baseline — not a sample.
        Compares against fraud rows in current production batch.
        Detects fraud pattern evolution specifically.
        This is the critical layer for fraud monitoring.

    Layer 3 — PR-AUC Tracking
        Directly measures model's ability to detect fraud.
        Primary metric for imbalanced classification.
        Fires retraining alert when performance degrades.
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)
        self.results = {}

    def run(self,
            ref_population: pd.DataFrame,
            current: pd.DataFrame,
            ref_fraud_history: pd.DataFrame = None) -> dict:
        """
        Parameters
        ----------
        ref_population    : Sample of training data for PSI (large, ~10k rows)
        current           : Current production batch
        ref_fraud_history : ALL historical fraud rows from training data.
                            If None, fraud rows from ref_population are used
                            (less reliable — smaller sample).
        """
        # Use full fraud history if provided, else fall back to sample fraud rows
        if ref_fraud_history is not None:
            ref_fraud = ref_fraud_history
            fraud_source = f'full history ({len(ref_fraud):,} rows)'
        else:
            ref_fraud = ref_population[ref_population[TARGET] == 1]
            fraud_source = f'population sample ({len(ref_fraud)} rows)'

        curr_fraud = current[current[TARGET] == 1]

        print(f'\n{"="*65}')
        print(f'  FRAUD DRIFT DETECTOR — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print(f'{"="*65}')
        print(f'Population reference : {ref_population.shape}')
        print(f'Current batch        : {current.shape} | Fraud rows: {len(curr_fraud)}')
        print(f'Fraud reference      : {fraud_source}')

        self.results = {
            'timestamp'        : datetime.now().isoformat(),
            'ref_population'   : len(ref_population),
            'current_size'     : len(current),
            'ref_fraud_rows'   : len(ref_fraud),
            'curr_fraud_rows'  : len(curr_fraud),
            'layer1_psi'       : self._layer1_psi(ref_population, current),
            'layer2_mw'        : self._layer2_mann_whitney(ref_fraud, curr_fraud),
            'layer3_performance': self._layer3_performance(current),
            'alerts'           : []
        }

        self._generate_alerts()
        self._print_summary()
        return self.results

    def _layer1_psi(self, ref: pd.DataFrame, curr: pd.DataFrame) -> dict:
        """Layer 1 — PSI on full population. Banking industry standard."""
        print(f'\n[Layer 1] PSI — Population Stability Index')
        print(f'  Thresholds: <0.10 stable | 0.10-0.20 moderate | >0.20 significant')
        results = {}
        for feat in MONITOR_FEATURES:
            r = compute_psi(ref[feat], curr[feat])
            results[feat] = r
            print(f'  {feat:<20} PSI={r["value"]:.4f}  [{r["level"]}]')
        return results

    def _layer2_mann_whitney(self,
                              ref_fraud: pd.DataFrame,
                              curr_fraud: pd.DataFrame) -> dict:
        """
        Layer 2 — Mann-Whitney U on fraud-only rows.
        ref_fraud = ALL historical fraud from training (7,506 rows)
        curr_fraud = fraud rows in current production batch (~54 rows)
        Unequal sizes are handled correctly by Mann-Whitney.
        """
        print(f'\n[Layer 2] Mann-Whitney U — fraud pattern comparison')
        print(f'  Reference fraud : {len(ref_fraud):,} rows (full training history)')
        print(f'  Current fraud   : {len(curr_fraud)} rows (production batch)')
        results = {}
        for feat in MONITOR_FEATURES:
            r = mann_whitney_test(ref_fraud[feat], curr_fraud[feat])
            results[feat] = {
                **r,
                'ref_median' : round(float(ref_fraud[feat].median()),  3),
                'curr_median': round(float(curr_fraud[feat].median()), 3),
                'change_pct' : round(
                    ((curr_fraud[feat].median() - ref_fraud[feat].median())
                     / ref_fraud[feat].median()) * 100, 1
                ) if ref_fraud[feat].median() != 0 else 0
            }
            if r['drifted'] is not None:
                status = 'DRIFT' if r['drifted'] else 'OK'
                print(f'  {feat:<20} p={r["p_value"]:.4f}'
                      f'  ref={results[feat]["ref_median"]}'
                      f'  curr={results[feat]["curr_median"]}'
                      f'  ({results[feat]["change_pct"]:+.1f}%)'
                      f'  [{status}]')
            else:
                print(f'  {feat:<20} [{r.get("note")}]')
        return results

    def _layer3_performance(self, current: pd.DataFrame) -> dict:
        """
        Layer 3 — PR-AUC and fraud-class metrics.
        PR-AUC is the primary metric for imbalanced fraud detection.
        Tracked here directly — not delegated to any external tool.
        """
        print(f'\n[Layer 3] Performance Metrics')
        X        = current.drop(columns=[TARGET])
        y        = current[TARGET]
        X_scaled = self.scaler.transform(X)
        y_prob   = self.model.predict_proba(X_scaled)[:, 1]
        y_pred   = self.model.predict(X_scaled)

        metrics = {
            'pr_auc'   : round(float(average_precision_score(y, y_prob)), 4),
            'roc_auc'  : round(float(roc_auc_score(y, y_prob)),           4),
            'f1_fraud' : round(float(f1_score(y, y_pred)),                4),
            'recall'   : round(float(recall_score(y, y_pred)),            4),
            'precision': round(float(precision_score(y, y_pred,
                                     zero_division=0)),                   4),
        }
        print(f'  PR-AUC    : {metrics["pr_auc"]}   ← primary metric')
        print(f'  ROC-AUC   : {metrics["roc_auc"]}')
        print(f'  F1-Fraud  : {metrics["f1_fraud"]}')
        print(f'  Recall    : {metrics["recall"]}')
        print(f'  Precision : {metrics["precision"]}')
        return metrics

    def _generate_alerts(self):
        alerts = []

        # PSI alerts
        for feat, r in self.results['layer1_psi'].items():
            if r['value'] >= PSI_MODERATE:
                alerts.append({
                    'level'  : 'HIGH',
                    'type'   : 'PSI_SIGNIFICANT_DRIFT',
                    'feature': feat,
                    'message': f"{feat} PSI={r['value']} — significant drift"
                })
            elif r['value'] >= PSI_NO_DRIFT:
                alerts.append({
                    'level'  : 'MEDIUM',
                    'type'   : 'PSI_MODERATE_DRIFT',
                    'feature': feat,
                    'message': f"{feat} PSI={r['value']} — moderate drift"
                })

        # Mann-Whitney fraud pattern alerts
        for feat, r in self.results['layer2_mw'].items():
            if r.get('drifted') and abs(r.get('change_pct', 0)) > 30:
                alerts.append({
                    'level'  : 'HIGH',
                    'type'   : 'FRAUD_PATTERN_DRIFT',
                    'feature': feat,
                    'message': (f"Fraud {feat} shifted {r['change_pct']:+.1f}% "
                                f"(p={r.get('p_value')})")
                })

        # PR-AUC performance alert
        pr_auc = self.results['layer3_performance']['pr_auc']
        if pr_auc < PRAUC_ALERT:
            alerts.append({
                'level'  : 'CRITICAL',
                'type'   : 'PERFORMANCE_DEGRADATION',
                'feature': 'pr_auc',
                'message': (f"PR-AUC={pr_auc} below threshold {PRAUC_ALERT}"
                            f" — model retraining required")
            })

        self.results['alerts'] = alerts

    def _print_summary(self):
        alerts = self.results['alerts']
        print(f'\n[Alerts] — {len(alerts)} raised')
        if alerts:
            for a in alerts:
                print(f'  [{a["level"]}] {a["type"]} : {a["message"]}')
        else:
            print('  No alerts — model within acceptable bounds')
        print(f'{"="*65}')

    def save_results(self, path: str = 'reports/drift_results.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f'Results saved → {path}')


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Drift Detector')
    parser.add_argument('--reference', default='data/processed/train_preprocessed.csv')
    parser.add_argument('--current',   default='data/processed/test_preprocessed.csv')
    parser.add_argument('--model',     default='models/xgboost_model.pkl')
    parser.add_argument('--scaler',    default='models/scaler.pkl')
    args = parser.parse_args()

    ref  = pd.read_csv(args.reference).select_dtypes(include=[np.number])
    curr = pd.read_csv(args.current).select_dtypes(include=[np.number])

    ref_sample        = ref.sample(n=min(10000, len(ref)), random_state=42)
    ref_fraud_history = ref[ref[TARGET] == 1]  # ALL fraud rows as baseline

    detector = FraudDriftDetector(args.model, args.scaler)
    detector.run(ref_sample, curr, ref_fraud_history=ref_fraud_history)
    detector.save_results()
