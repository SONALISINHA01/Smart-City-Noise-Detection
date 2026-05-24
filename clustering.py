"""
================================================================================
  Clustering Module
  K-Means · DBSCAN · Gaussian Mixture Model
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  K-Means
# ─────────────────────────────────────────────────────────────────────────────

class KMeansClustering:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=15,
            max_iter=500,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray):
        self.labels_ = self.model.fit_predict(X)
        self.inertia_ = self.model.inertia_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def elbow_curve(self, X: np.ndarray, k_range=range(2, 12)) -> dict:
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            lbl = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, lbl, sample_size=2000))
        return {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}

    def save(self, path="models/kmeans.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path="models/kmeans.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.n_clusters = obj.model.n_clusters
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

class DBSCANClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

    def fit(self, X: np.ndarray):
        self.labels_ = self.model.fit_predict(X)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise    = (self.labels_ == -1).sum()
        print(f"  DBSCAN → clusters: {n_clusters}  |  noise points: {n_noise}")
        return self

    def save(self, path="models/dbscan.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path="models/dbscan.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.eps = obj.model.eps
        obj.min_samples = obj.model.min_samples
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian Mixture Model
# ─────────────────────────────────────────────────────────────────────────────

class GMMClustering:
    def __init__(self, n_components: int = 5, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=300,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.bic_     = self.model.bic(X)
        self.aic_     = self.model.aic(X)
        print(f"  GMM → BIC: {self.bic_:.2f}  |  AIC: {self.aic_:.2f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path="models/gmm.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path="models/gmm.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.n_components = obj.model.n_components
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, model_name: str = "") -> dict:
    mask = labels != -1    # exclude DBSCAN noise for scoring
    valid_X   = X[mask]
    valid_lbl = labels[mask]
    n_clusters = len(set(valid_lbl))

    if n_clusters < 2:
        print(f"  ⚠️  {model_name}: only {n_clusters} cluster(s) – skipping scores")
        return {"model": model_name, "n_clusters": n_clusters}

    sil = silhouette_score(valid_X, valid_lbl, sample_size=min(3000, len(valid_X)))
    dbi = davies_bouldin_score(valid_X, valid_lbl)
    chi = calinski_harabasz_score(valid_X, valid_lbl)

    metrics = {
        "model":              model_name,
        "n_clusters":         n_clusters,
        "silhouette_score":   round(sil, 4),
        "davies_bouldin":     round(dbi, 4),
        "calinski_harabasz":  round(chi, 2),
        "noise_points":       int((labels == -1).sum()),
    }
    print(f"  📊 {model_name:20s} | Sil={sil:.3f}  DBI={dbi:.3f}  CHI={chi:.1f}")
    return metrics


def assign_noise_severity(labels: np.ndarray, df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Map cluster IDs → severity label based on median dB per cluster."""
    df = df.copy()
    df[cluster_col] = labels

    cluster_db = df.groupby(cluster_col)["avg_decibels"].median().sort_values()
    n = len(cluster_db)
    severity = {}
    for rank, (cid, _) in enumerate(cluster_db.items()):
        pct = rank / max(n - 1, 1)
        if   pct < 0.2:  severity[cid] = "Very Low"
        elif pct < 0.4:  severity[cid] = "Low"
        elif pct < 0.6:  severity[cid] = "Moderate"
        elif pct < 0.8:  severity[cid] = "High"
        else:            severity[cid] = "Extreme"
    severity[-1] = "Anomaly"   # DBSCAN noise points

    df["noise_severity"] = df[cluster_col].map(severity)
    return df
