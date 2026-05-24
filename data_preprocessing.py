"""
================================================================================
  Data Preprocessing & Feature Engineering
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
import os

# Features used for unsupervised learning (no labels)
FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend",
    "weather_code", "temperature_c", "humidity_pct", "wind_speed_kmh",
    "traffic_density", "vehicle_count", "honking_frequency",
    "heavy_vehicle_pct", "industrial_activity", "construction_activity",
    "population_density", "event_activity", "avg_decibels",
]

# Spatial features kept separate for map viz
SPATIAL_COLS = ["latitude", "longitude"]


class NoisePollutionPreprocessor:
    """End-to-end preprocessing pipeline for the smart-city noise dataset."""

    def __init__(self, scaler_path: str = "models/scaler.pkl"):
        self.scaler      = RobustScaler()
        self.scaler_path = scaler_path
        self.feature_cols = FEATURE_COLS

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._engineer_features(df)
        X  = df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        self._save_scaler()
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._engineer_features(df)
        X  = df[self.feature_cols].values
        return self.scaler.transform(X)

    def load_scaler(self):
        self.scaler = joblib.load(self.scaler_path)
        return self

    # ── Feature engineering ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Cyclical time encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Composite activity index
        df["urban_activity_index"] = (
            0.3 * df["traffic_density"] / 200 +
            0.2 * df["industrial_activity"] / 100 +
            0.2 * df["construction_activity"] / 100 +
            0.15 * df["honking_frequency"] / 100 +
            0.15 * df["population_density"] / 25000
        ) * 100

        # Noise-per-vehicle ratio (proxy for aggressive driving)
        df["noise_per_vehicle"] = np.where(
            df["vehicle_count"] > 0,
            df["avg_decibels"] / (df["vehicle_count"] + 1),
            0
        )

        # Peak-hour flag
        df["is_peak_hour"] = df["hour"].apply(
            lambda h: 1 if (7 <= h < 10 or 17 <= h < 20) else 0
        )

        # Update feature list to include engineered features
        extra = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "urban_activity_index", "noise_per_vehicle", "is_peak_hour"
        ]
        for col in extra:
            if col not in self.feature_cols:
                self.feature_cols.append(col)

        return df

    # ── Utility ───────────────────────────────────────────────────────────────

    def _save_scaler(self):
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"  ✅ Scaler saved → {self.scaler_path}")

    @staticmethod
    def load_raw(path: str = "data/smart_city_noise.csv") -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"  📂 Loaded {path}  |  Shape: {df.shape}")
        return df

    @staticmethod
    def summary(df: pd.DataFrame):
        print("\n" + "="*60)
        print("  DATASET SUMMARY")
        print("="*60)
        print(f"  Rows        : {len(df):,}")
        print(f"  Columns     : {df.shape[1]}")
        print(f"  Time range  : {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"  dB range    : {df['avg_decibels'].min():.1f} – {df['avg_decibels'].max():.1f}")
        print(f"  Zone types  : {df['zone_type'].value_counts().to_dict()}")
        print(f"  Missing vals: {df.isnull().sum().sum()}")
        print("="*60)


def add_noise_severity_label(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based severity for evaluation / colour-coding (NOT used in training)."""
    bins   = [0, 50, 60, 70, 85, 200]
    labels = ["Very Low", "Low", "Moderate", "High", "Extreme"]
    df["noise_severity"] = pd.cut(
        df["avg_decibels"], bins=bins, labels=labels, right=False
    )
    return df
