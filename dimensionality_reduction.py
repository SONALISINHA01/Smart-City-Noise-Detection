"""
================================================================================
  Dimensionality Reduction & Pattern Discovery
  PCA · t-SNE · UMAP (optional)
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib, os
import warnings
warnings.filterwarnings("ignore")


class DimensionalityReducer:
    """PCA + t-SNE pipeline for 2D / 3D visualisation and feature insight."""

    def __init__(self, n_components_pca: int = 10, random_state: int = 42):
        self.n_components_pca = n_components_pca
        self.random_state      = random_state
        self.pca_full_         = None    # fitted on full feature set
        self.pca_2d_           = None    # 2-component PCA for quick viz
        self.pca_3d_           = None

    # ── PCA ───────────────────────────────────────────────────────────────────

    def fit_pca(self, X: np.ndarray):
        """Fit a full PCA to get variance explained + a 2-D/3-D reduction."""
        self.pca_full_ = PCA(n_components=min(self.n_components_pca, X.shape[1]),
                             random_state=self.random_state)
        self.pca_full_.fit(X)

        self.pca_2d_ = PCA(n_components=2, random_state=self.random_state).fit(X)
        self.pca_3d_ = PCA(n_components=3, random_state=self.random_state).fit(X)
        return self

    def transform_pca(self, X: np.ndarray, dims: int = 2) -> np.ndarray:
        if dims == 2:
            return self.pca_2d_.transform(X)
        elif dims == 3:
            return self.pca_3d_.transform(X)
        else:
            return self.pca_full_.transform(X)

    def variance_explained(self) -> dict:
        evr = self.pca_full_.explained_variance_ratio_
        return {
            "per_component":      list(evr),
            "cumulative":         list(np.cumsum(evr)),
            "n_components_95pct": int(np.searchsorted(np.cumsum(evr), 0.95)) + 1,
        }

    def feature_importance_pca(self, feature_names: list) -> pd.DataFrame:
        """Absolute loading of each feature on PC1 & PC2."""
        loadings = self.pca_full_.components_[:2]
        df = pd.DataFrame(
            np.abs(loadings).T,
            index=feature_names,
            columns=["PC1_loading", "PC2_loading"],
        )
        df["total_importance"] = df.sum(axis=1)
        return df.sort_values("total_importance", ascending=False)

    # ── t-SNE ─────────────────────────────────────────────────────────────────

    def tsne_2d(self, X: np.ndarray, perplexity: int = 40, n_iter: int = 1000) -> np.ndarray:
        """Run t-SNE on a PCA-reduced version for speed (recommended ≤ 5000 pts)."""
        if X.shape[0] > 5000:
            idx = np.random.choice(X.shape[0], 5000, replace=False)
            X = X[idx]

        # First reduce to 30 dims with PCA for t-SNE stability
        n_pc = min(30, X.shape[1])
        X_pca = PCA(n_components=n_pc, random_state=self.random_state).fit_transform(X)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate="auto",
            init="pca",
            random_state=self.random_state,
            n_jobs=-1,
        )
        return tsne.fit_transform(X_pca)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, dir_path: str = "models"):
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.pca_full_, os.path.join(dir_path, "pca_full.pkl"))
        joblib.dump(self.pca_2d_,   os.path.join(dir_path, "pca_2d.pkl"))
        joblib.dump(self.pca_3d_,   os.path.join(dir_path, "pca_3d.pkl"))
        print(f"  ✅ PCA models saved → {dir_path}/")

    @classmethod
    def load(cls, dir_path: str = "models"):
        obj = cls.__new__(cls)
        obj.pca_full_ = joblib.load(os.path.join(dir_path, "pca_full.pkl"))
        obj.pca_2d_   = joblib.load(os.path.join(dir_path, "pca_2d.pkl"))
        obj.pca_3d_   = joblib.load(os.path.join(dir_path, "pca_3d.pkl"))
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Pattern Discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def peak_noise_hours(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Average dB by hour – returns sorted DataFrame."""
    return (
        df.groupby("hour")["avg_decibels"]
        .mean()
        .reset_index()
        .rename(columns={"avg_decibels": "mean_db"})
        .sort_values("mean_db", ascending=False)
        .head(top_n)
    )


def weekly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    return (
        df.groupby("day_of_week")["avg_decibels"]
        .mean()
        .reset_index()
        .assign(day_name=lambda d: d["day_of_week"].map(day_map))
    )


def zone_activity_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "zone_type", "avg_decibels", "traffic_density",
        "industrial_activity", "construction_activity",
        "honking_frequency", "population_density",
    ]
    return df[cols].groupby("zone_type").mean().round(2)


def generate_smart_recommendations(df: pd.DataFrame, anomaly_col: str = "if_anomaly") -> list:
    recs = []
    high_noise_zones = df[df["avg_decibels"] > 80]["zone_type"].value_counts()
    for zone, cnt in high_noise_zones.items():
        recs.append(f"🔇 {zone}: {cnt} high-noise readings detected – deploy noise barriers.")

    peak = peak_noise_hours(df, top_n=2)
    for _, row in peak.iterrows():
        recs.append(f"⏰ Hour {int(row['hour']):02d}:00 – average {row['mean_db']:.1f} dB, consider traffic diversion.")

    if anomaly_col in df.columns:
        n_anom = (df[anomaly_col] == -1).sum()
        recs.append(f"🚨 {n_anom} anomalous noise events detected – investigate immediately.")

    avg_db = df["avg_decibels"].mean()
    if avg_db > 70:
        recs.append(f"📢 City-wide average noise ({avg_db:.1f} dB) exceeds WHO limit of 70 dB.")
    elif avg_db > 55:
        recs.append(f"⚠️  City-wide average noise ({avg_db:.1f} dB) approaching concerning levels.")

    return recs
