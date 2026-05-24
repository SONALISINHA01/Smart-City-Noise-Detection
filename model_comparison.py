"""
================================================================================
  Model Comparison & Evaluation Module
  KMeans · DBSCAN · Isolation Forest · Autoencoder
  Metrics: Silhouette · DBI · CH Index · Reconstruction Error · Anomaly AUC
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Clustering Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clustering(X: np.ndarray, labels: np.ndarray,
                         model_name: str = "Model") -> dict:
    """Compute standard unsupervised clustering quality metrics."""
    valid = labels != -1     # exclude DBSCAN noise
    n_clusters = len(set(labels[valid]))

    if n_clusters < 2 or valid.sum() < 50:
        return {
            "model":       model_name,
            "n_clusters":  n_clusters,
            "silhouette":  None,
            "dbi":         None,
            "ch_index":    None,
            "note":        "Insufficient clusters / samples for metrics",
        }

    X_v, lbl_v = X[valid], labels[valid]
    sil = round(float(silhouette_score(X_v, lbl_v, sample_size=min(3000, len(X_v)))), 4)
    dbi = round(float(davies_bouldin_score(X_v, lbl_v)), 4)
    ch  = round(float(calinski_harabasz_score(X_v, lbl_v)), 2)

    return {
        "model":      model_name,
        "n_clusters": n_clusters,
        "silhouette": sil,   # higher → better (max 1)
        "dbi":        dbi,   # lower  → better
        "ch_index":   ch,    # higher → better
        "n_noise":    int((labels == -1).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly Detection Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_anomaly_detector(
    scores:    np.ndarray,          # continuous anomaly scores [0,1]
    pred_bin:  np.ndarray,          # binary predictions (1=anomaly)
    true_bin:  np.ndarray,          # ground-truth labels (1=anomaly)
    model_name: str = "Detector",
) -> dict:
    """Evaluate anomaly detection quality against ground-truth injected labels."""
    if true_bin.sum() == 0:
        return {"model": model_name, "note": "No ground-truth anomalies found"}

    try:
        auc = round(float(roc_auc_score(true_bin, scores)), 4)
    except Exception:
        auc = None

    prec = round(float(precision_score(true_bin, pred_bin, zero_division=0)), 4)
    rec  = round(float(recall_score(true_bin, pred_bin, zero_division=0)), 4)
    f1   = round(float(f1_score(true_bin, pred_bin, zero_division=0)), 4)

    return {
        "model":        model_name,
        "roc_auc":      auc,
        "precision":    prec,
        "recall":       rec,
        "f1_score":     f1,
        "n_predicted":  int(pred_bin.sum()),
        "n_true":       int(true_bin.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Autoencoder-specific
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_autoencoder(reconstruction_errors: np.ndarray,
                          true_bin: np.ndarray,
                          threshold: float,
                          model_name: str = "Autoencoder") -> dict:
    pred_bin = (reconstruction_errors > threshold).astype(int)
    scores   = (reconstruction_errors - reconstruction_errors.min()) / \
               (reconstruction_errors.max() - reconstruction_errors.min() + 1e-9)
    base = evaluate_anomaly_detector(scores, pred_bin, true_bin, model_name)
    base["mean_recon_error"] = round(float(reconstruction_errors.mean()), 6)
    base["threshold"]        = round(float(threshold), 6)
    return base


# ─────────────────────────────────────────────────────────────────────────────
#  Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(clustering_results: list,
                             anomaly_results: list) -> pd.DataFrame:
    """Merge clustering and anomaly metrics into one comparison DataFrame."""
    rows = []

    for c in clustering_results:
        rows.append({
            "Model":          c.get("model", ""),
            "Type":           "Clustering",
            "Silhouette ↑":   c.get("silhouette"),
            "DBI ↓":          c.get("dbi"),
            "CH Index ↑":     c.get("ch_index"),
            "ROC-AUC ↑":      None,
            "F1 Score ↑":     None,
            "n_clusters":     c.get("n_clusters"),
        })

    for a in anomaly_results:
        rows.append({
            "Model":          a.get("model", ""),
            "Type":           "Anomaly Detection",
            "Silhouette ↑":   None,
            "DBI ↓":          None,
            "CH Index ↑":     None,
            "ROC-AUC ↑":      a.get("roc_auc"),
            "F1 Score ↑":     a.get("f1_score"),
            "n_clusters":     None,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Radar-chart data builder
# ─────────────────────────────────────────────────────────────────────────────

def build_radar_metrics(clustering_results: list) -> dict:
    """
    Returns dict suitable for visualizations.plot_model_comparison().
    Normalises DBI so higher = better.
    """
    radar = {}
    max_dbi = max((r.get("dbi") or 0) for r in clustering_results) + 1e-9
    for r in clustering_results:
        model = r.get("model", "?")
        radar[model] = {
            "Silhouette": r.get("silhouette") or 0,
            "1/DBI":      round(1 / (r.get("dbi") or max_dbi), 4),
            "CH/1000":    round((r.get("ch_index") or 0) / 1000, 4),
        }
    return radar
