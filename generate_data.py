"""
================================================================================
  Smart City Noise Pollution Dataset Generator
  Generates realistic synthetic data with spatial & temporal patterns
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

SEED = 42
np.random.seed(SEED)

# ─── City configuration ──────────────────────────────────────────────────────
CITY_CENTER_LAT = 28.6139   # New Delhi-inspired smart city
CITY_CENTER_LON = 77.2090
N_SAMPLES       = 5000
START_DATE      = datetime(2024, 1, 1)

# ─── Zone definitions (lat/lon offsets, multipliers) ─────────────────────────
ZONES = {
    "Industrial":     {"lat_off":  0.05, "lon_off":  0.06, "noise_mul": 1.6, "traffic_mul": 1.4},
    "Commercial_CBD": {"lat_off":  0.00, "lon_off":  0.00, "noise_mul": 1.4, "traffic_mul": 1.6},
    "Residential":    {"lat_off": -0.04, "lon_off": -0.05, "noise_mul": 0.7, "traffic_mul": 0.6},
    "Transport_Hub":  {"lat_off":  0.03, "lon_off": -0.04, "noise_mul": 1.5, "traffic_mul": 1.8},
    "Green_Zone":     {"lat_off": -0.06, "lon_off":  0.04, "noise_mul": 0.4, "traffic_mul": 0.3},
    "Suburban":       {"lat_off": -0.07, "lon_off": -0.07, "noise_mul": 0.6, "traffic_mul": 0.5},
}

WEATHER_MAP = {0: "Clear", 1: "Cloudy", 2: "Rainy", 3: "Foggy", 4: "Windy"}


def _time_multiplier(hour: int) -> float:
    """Rush-hour & night-time noise profile."""
    if   7  <= hour < 10: return 1.5   # Morning rush
    elif 10 <= hour < 17: return 1.2   # Daytime
    elif 17 <= hour < 20: return 1.6   # Evening rush
    elif 20 <= hour < 23: return 0.9   # Evening
    else:                 return 0.5   # Night


def _weather_noise_effect(weather: int) -> float:
    return {0: 1.0, 1: 0.95, 2: 0.75, 3: 0.8, 4: 1.1}.get(weather, 1.0)


def generate_dataset(n_samples: int = N_SAMPLES, save_path: str = "data/smart_city_noise.csv") -> pd.DataFrame:
    records = []
    zone_names   = list(ZONES.keys())
    zone_weights = [0.15, 0.20, 0.25, 0.12, 0.10, 0.18]   # sampling probability

    for i in range(n_samples):
        # ── Pick zone ────────────────────────────────────────────────────────
        zone = np.random.choice(zone_names, p=zone_weights)
        z    = ZONES[zone]

        # ── Timestamp ────────────────────────────────────────────────────────
        ts   = START_DATE + timedelta(
            days   = np.random.randint(0, 180),
            hours  = np.random.randint(0, 24),
            minutes= np.random.randint(0, 60),
        )
        hour        = ts.hour
        day_of_week = ts.weekday()          # 0=Mon … 6=Sun
        is_weekend  = int(day_of_week >= 5)

        # ── GPS with Gaussian scatter inside zone ─────────────────────────────
        lat = CITY_CENTER_LAT + z["lat_off"] + np.random.normal(0, 0.015)
        lon = CITY_CENTER_LON + z["lon_off"] + np.random.normal(0, 0.015)

        # ── Weather ───────────────────────────────────────────────────────────
        weather_code = np.random.choice([0,1,2,3,4], p=[0.40,0.25,0.15,0.10,0.10])
        temperature  = np.random.normal(25, 8)
        humidity     = np.random.uniform(30, 95)
        wind_speed   = np.random.exponential(10)

        # ── Traffic features ──────────────────────────────────────────────────
        t_mul = _time_multiplier(hour) * z["traffic_mul"] * (1.3 if is_weekend and zone=="Commercial_CBD" else 1.0)
        traffic_density  = np.clip(np.random.normal(60, 20) * t_mul, 0, 200)
        vehicle_count    = np.clip(np.random.poisson(120 * t_mul), 0, 500).astype(int)
        honking_freq     = np.clip(np.random.exponential(15) * z["traffic_mul"] * t_mul, 0, 100)
        heavy_vehicle_pct= np.clip(np.random.beta(2, 5) * 100, 0, 100)

        # ── Urban activity ────────────────────────────────────────────────────
        industrial_activity  = np.clip(np.random.beta(2, 3) * 100 * (2.0 if zone == "Industrial" else 0.3), 0, 100)
        construction_activity= np.clip(np.random.beta(1.5, 4) * 100 * z["noise_mul"], 0, 100)
        population_density   = np.clip(np.random.normal(8000, 3000) * z["noise_mul"], 500, 25000)
        event_activity       = int(np.random.random() < (0.1 if zone in ["Commercial_CBD","Transport_Hub"] else 0.03))

        # ── Composite noise (dB) ──────────────────────────────────────────────
        base_db       = 45.0
        traffic_noise = 0.08 * traffic_density + 0.03 * vehicle_count
        horn_noise    = 0.15 * honking_freq
        industry_noise= 0.12 * industrial_activity
        construct_noise=0.10 * construction_activity
        pop_noise     = 0.001 * population_density
        weather_adj   = _weather_noise_effect(weather_code)
        time_adj      = _time_multiplier(hour)
        event_bonus   = 8 * event_activity
        zone_base     = z["noise_mul"] * 15

        avg_decibels = (
            base_db + traffic_noise + horn_noise +
            industry_noise + construct_noise + pop_noise +
            zone_base + event_bonus
        ) * weather_adj * time_adj + np.random.normal(0, 3)
        avg_decibels = np.clip(avg_decibels, 30, 115)

        # ── Inject 3 % synthetic anomalies ───────────────────────────────────
        is_anomaly = 0
        if np.random.random() < 0.03:
            avg_decibels = np.clip(avg_decibels + np.random.uniform(20, 35), 90, 130)
            traffic_density *= np.random.uniform(1.5, 2.5)
            honking_freq    *= np.random.uniform(2, 4)
            is_anomaly       = 1          # kept for evaluation only, not used in training

        records.append({
            "timestamp":           ts,
            "hour":                hour,
            "day_of_week":         day_of_week,
            "is_weekend":          is_weekend,
            "latitude":            round(lat, 6),
            "longitude":           round(lon, 6),
            "zone_type":           zone,           # for validation, not fed to model
            "weather_code":        weather_code,
            "weather_condition":   WEATHER_MAP[weather_code],
            "temperature_c":       round(temperature, 2),
            "humidity_pct":        round(humidity, 2),
            "wind_speed_kmh":      round(wind_speed, 2),
            "traffic_density":     round(traffic_density, 2),
            "vehicle_count":       int(vehicle_count),
            "honking_frequency":   round(honking_freq, 2),
            "heavy_vehicle_pct":   round(heavy_vehicle_pct, 2),
            "industrial_activity": round(industrial_activity, 2),
            "construction_activity":round(construction_activity, 2),
            "population_density":  round(population_density, 2),
            "event_activity":      event_activity,
            "avg_decibels":        round(avg_decibels, 2),
            "is_anomaly_label":    is_anomaly,     # hidden ground-truth, not for training
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved → {save_path}  |  Shape: {df.shape}")
    print(f"   Anomalies injected: {df['is_anomaly_label'].sum()} / {len(df)}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe())
