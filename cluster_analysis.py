"""
================================================================================
  Cluster Analysis Module
  Statistical profiling of each cluster + zone identification
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Zone identification thresholds ────────────────────────────────────────────
ZONE_RULES = {
    "Industrial Zone":      {"industrial_activity": (">", 40), "avg_decibels": (">", 70)},
    "Traffic-Heavy Zone":   {"traffic_density":     (">", 80), "honking_frequency": (">", 20)},
    "Construction Zone":    {"construction_activity": (">", 40)},
    "Transport Hub":        {"vehicle_count":        (">", 150), "traffic_density": (">", 70)},
    "Residential Quiet Zone":{"avg_decibels":        ("<", 55), "population_density": ("<", 6000)},
    "Commercial Zone":      {"traffic_density":      (">", 60), "population_density": (">", 8000)},
    "Green/Park Zone":      {"avg_decibels":         ("<", 50), "industrial_activity": ("<", 15)},
}

NOISE_SEVERITY_BINS   = [0,  50,  60,  70,  85, 200]
NOISE_SEVERITY_LABELS = ["Very Low", "Low", "Moderate", "High", "Extreme"]


# ─────────────────────────────────────────────────────────────────────────────
class ClusterAnalyzer:
    """Full statistical analysis and labelling for K-Means / DBSCAN clusters."""

    def __init__(self, df: pd.DataFrame, cluster_col: str = "kmeans_cluster"):
        self.df          = df.copy()
        self.cluster_col = cluster_col
        self.summary_df: Optional[pd.DataFrame] = None
        self.zone_map:   Dict[int, str]          = {}

    # ── Core summary ──────────────────────────────────────────────────────────

    def compute_summary(self) -> pd.DataFrame:
        """Aggregate statistics per cluster."""
        agg_cols = {
            "avg_decibels":          ["mean", "std", "min", "max"],
            "traffic_density":       "mean",
            "vehicle_count":         "mean",
            "honking_frequency":     "mean",
            "industrial_activity":   "mean",
            "construction_activity": "mean",
            "population_density":    "mean",
            "is_peak_hour":          "mean",
            "event_activity":        "sum",
        }
        # Only include columns that exist
        agg_cols = {k: v for k, v in agg_cols.items() if k in self.df.columns}

        summary = self.df.groupby(self.cluster_col).agg(agg_cols)
        summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                           for c in summary.columns]
        summary["count"] = self.df[self.cluster_col].value_counts()

        # Noise severity percentage
        if "noise_severity" in self.df.columns:
            high_pct = (
                self.df[self.df["noise_severity"].isin(["High", "Extreme"])]
                .groupby(self.cluster_col)
                .size()
                / self.df.groupby(self.cluster_col).size()
                * 100
            )
            summary["high_noise_pct"] = high_pct.round(1)

        summary = summary.round(2).reset_index()
        summary.rename(columns={self.cluster_col: "cluster_id"}, inplace=True)

        # Zone label
        summary["zone_label"] = summary["cluster_id"].apply(self._identify_zone)
        self.summary_df = summary
        return summary

    # ── Zone identification ───────────────────────────────────────────────────

    def _identify_zone(self, cluster_id: int) -> str:
        if cluster_id == -1:
            return "Noise Anomaly"
        row = self.df[self.df[self.cluster_col] == cluster_id].mean(numeric_only=True)
        scores: Dict[str, int] = {}

        for zone_name, rules in ZONE_RULES.items():
            score = 0
            for feat, (op, thresh) in rules.items():
                if feat not in row.index:
                    continue
                val = row[feat]
                if op == ">" and val > thresh:
                    score += 1
                elif op == "<" and val < thresh:
                    score += 1
            scores[zone_name] = score

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Mixed Zone"

    def build_zone_map(self) -> Dict[int, str]:
        if self.summary_df is None:
            self.compute_summary()
        self.zone_map = dict(zip(
            self.summary_df["cluster_id"],
            self.summary_df["zone_label"]
        ))
        return self.zone_map

    # ── Risk scoring ──────────────────────────────────────────────────────────

    def risk_score(self) -> pd.DataFrame:
        """0-100 risk score per cluster based on noise + activity."""
        if self.summary_df is None:
            self.compute_summary()
        s = self.summary_df.copy()
        db_col = "avg_decibels_mean" if "avg_decibels_mean" in s.columns else "avg_decibels"
        td_col = "traffic_density_mean" if "traffic_density_mean" in s.columns else "traffic_density"
        ia_col = "industrial_activity_mean" if "industrial_activity_mean" in s.columns else "industrial_activity"

        s["risk_score"] = (
            0.45 * s[db_col].clip(30, 110).apply(lambda x: (x-30)/80*100) +
            0.25 * s.get(td_col, pd.Series(0, index=s.index)).clip(0, 200).apply(lambda x: x/2) +
            0.20 * s.get(ia_col, pd.Series(0, index=s.index)).clip(0, 100) +
            0.10 * s.get("high_noise_pct", pd.Series(0, index=s.index))
        ).round(1)

        s["risk_level"] = s["risk_score"].apply(
            lambda x: "🔴 Critical" if x >= 75 else (
                      "🟠 High"     if x >= 55 else (
                      "🟡 Moderate" if x >= 35 else "🟢 Low"))
        )
        return s[["cluster_id", "zone_label", "risk_score", "risk_level"]]

    # ── Peak hour analysis ────────────────────────────────────────────────────

    def peak_hour_analysis(self) -> pd.DataFrame:
        return (
            self.df.groupby([self.cluster_col, "hour"])["avg_decibels"]
            .mean()
            .reset_index()
            .rename(columns={"avg_decibels": "mean_db", self.cluster_col: "cluster_id"})
            .sort_values(["cluster_id", "mean_db"], ascending=[True, False])
        )

    # ── Recommendations ───────────────────────────────────────────────────────

    def generate_recommendations(self) -> List[str]:
        if self.summary_df is None:
            self.compute_summary()
        recs: List[str] = []
        db_col = "avg_decibels_mean" if "avg_decibels_mean" in self.summary_df.columns else "avg_decibels"
        td_col = "traffic_density_mean" if "traffic_density_mean" in self.summary_df.columns else "traffic_density"
        ia_col = "industrial_activity_mean" if "industrial_activity_mean" in self.summary_df.columns else "industrial_activity"

        for _, row in self.summary_df.iterrows():
            label = row["zone_label"]
            cid   = row["cluster_id"]
            db    = row.get(db_col, 0)
            if db > 80:
                recs.append(f"🔇 Cluster {cid} ({label}): Install sound-absorbing barriers; avg dB={db:.1f}.")
            if row.get(td_col, 0) > 100:
                recs.append(f"🚦 Cluster {cid} ({label}): Implement traffic flow management / signal optimisation.")
            if row.get(ia_col, 0) > 50:
                recs.append(f"🏭 Cluster {cid} ({label}): Enforce industrial noise curfews (night hours).")
        if not recs:
            recs.append("✅ All zones within acceptable noise thresholds.")
        return recs

    # ── Pretty print ──────────────────────────────────────────────────────────

    def print_report(self):
        if self.summary_df is None:
            self.compute_summary()
        print("\n" + "="*70)
        print("  CLUSTER ANALYSIS REPORT")
        print("="*70)
        print(self.summary_df.to_string(index=False))
        print("\n--- RECOMMENDATIONS ---")
        for r in self.generate_recommendations():
            print(" ", r)
        print("="*70)
