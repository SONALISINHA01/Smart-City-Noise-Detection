"""
================================================================================
  Real-Time Noise Simulation Module
  Streams synthetic city noise data · Live anomaly detection
================================================================================
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
from datetime import datetime
from typing import Callable, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Zone definitions (mirrors dataset generator) ─────────────────────────────
ZONES = {
    "Industrial":     {"lat_off":  0.05, "lon_off":  0.06, "noise_mul": 1.6, "traffic_mul": 1.4},
    "Commercial_CBD": {"lat_off":  0.00, "lon_off":  0.00, "noise_mul": 1.4, "traffic_mul": 1.6},
    "Residential":    {"lat_off": -0.04, "lon_off": -0.05, "noise_mul": 0.7, "traffic_mul": 0.6},
    "Transport_Hub":  {"lat_off":  0.03, "lon_off": -0.04, "noise_mul": 1.5, "traffic_mul": 1.8},
    "Green_Zone":     {"lat_off": -0.06, "lon_off":  0.04, "noise_mul": 0.4, "traffic_mul": 0.3},
    "Suburban":       {"lat_off": -0.07, "lon_off": -0.07, "noise_mul": 0.6, "traffic_mul": 0.5},
}
CITY_CENTER = (28.6139, 77.2090)
WEATHER_MAP = {0: "Clear", 1: "Cloudy", 2: "Rainy", 3: "Foggy", 4: "Windy"}


def _time_mul(hour: int) -> float:
    if   7  <= hour < 10: return 1.5
    elif 10 <= hour < 17: return 1.2
    elif 17 <= hour < 20: return 1.6
    elif 20 <= hour < 23: return 0.9
    else:                 return 0.5


def _weather_effect(w: int) -> float:
    return {0: 1.0, 1: 0.95, 2: 0.75, 3: 0.8, 4: 1.1}.get(w, 1.0)


# ─────────────────────────────────────────────────────────────────────────────

class RealTimeSimulator:
    """
    Generates a continuous stream of synthetic noise sensor readings.
    Optionally injects spike anomalies every N readings for realism.
    """

    def __init__(
        self,
        emit_interval: float = 1.0,     # seconds between readings
        anomaly_rate:  float = 0.05,    # probability of injecting an anomaly
        seed: Optional[int] = None,
    ):
        self.emit_interval = emit_interval
        self.anomaly_rate  = anomaly_rate
        self._rng          = np.random.default_rng(seed)
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self.data_queue: queue.Queue = queue.Queue(maxsize=500)
        self._history: list = []

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, callback: Optional[Callable] = None):
        """Start background streaming thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._stream_loop, args=(callback,), daemon=True
        )
        self._thread.start()
        print("▶  Real-time simulator started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("⏹  Real-time simulator stopped.")

    def get_latest(self, n: int = 50) -> pd.DataFrame:
        """Return last n records as a DataFrame."""
        return pd.DataFrame(self._history[-n:]) if self._history else pd.DataFrame()

    def generate_single(self, hour: Optional[int] = None,
                         zone: Optional[str] = None,
                         force_anomaly: bool = False) -> dict:
        """Generate one synthetic reading (useful for manual simulation in dashboard)."""
        return self._generate_record(hour=hour, zone=zone,
                                     force_anomaly=force_anomaly)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _stream_loop(self, callback):
        while self._running:
            record = self._generate_record()
            self._history.append(record)
            try:
                self.data_queue.put_nowait(record)
            except queue.Full:
                self.data_queue.get_nowait()   # drop oldest
                self.data_queue.put_nowait(record)
            if callback:
                callback(record)
            time.sleep(self.emit_interval)

    def _generate_record(self, hour=None, zone=None, force_anomaly=False) -> dict:
        now        = datetime.now()
        hour       = hour if hour is not None else now.hour
        zone_names = list(ZONES.keys())
        zone       = zone if zone else self._rng.choice(zone_names)
        z          = ZONES[zone]

        weather_code = int(self._rng.choice([0,1,2,3,4], p=[0.40,0.25,0.15,0.10,0.10]))
        is_weekend   = int(now.weekday() >= 5)
        t_mul        = _time_mul(hour) * z["traffic_mul"]

        lat = CITY_CENTER[0] + z["lat_off"] + float(self._rng.normal(0, 0.012))
        lon = CITY_CENTER[1] + z["lon_off"] + float(self._rng.normal(0, 0.012))

        traffic_density      = float(np.clip(self._rng.normal(60, 20) * t_mul, 0, 200))
        vehicle_count        = int(np.clip(self._rng.poisson(120 * t_mul), 0, 500))
        honking_freq         = float(np.clip(self._rng.exponential(15) * z["traffic_mul"] * t_mul, 0, 100))
        industrial_activity  = float(np.clip(self._rng.beta(2, 3) * 100 *
                                              (2.0 if zone == "Industrial" else 0.3), 0, 100))
        construction_activity= float(np.clip(self._rng.beta(1.5, 4) * 100 * z["noise_mul"], 0, 100))
        population_density   = float(np.clip(self._rng.normal(8000, 3000) * z["noise_mul"], 500, 25000))
        temperature_c        = float(self._rng.normal(25, 8))
        humidity_pct         = float(self._rng.uniform(30, 95))
        wind_speed_kmh       = float(self._rng.exponential(10))
        heavy_vehicle_pct    = float(np.clip(self._rng.beta(2, 5) * 100, 0, 100))
        event_activity       = int(self._rng.random() < 0.05)

        base_db = (
            45.0
            + 0.08 * traffic_density
            + 0.03 * vehicle_count
            + 0.15 * honking_freq
            + 0.12 * industrial_activity
            + 0.10 * construction_activity
            + 0.001 * population_density
            + z["noise_mul"] * 15
            + 8 * event_activity
        ) * _weather_effect(weather_code) * _time_mul(hour) + float(self._rng.normal(0, 3))
        avg_decibels = float(np.clip(base_db, 30, 115))

        is_anomaly = False
        if force_anomaly or (self._rng.random() < self.anomaly_rate):
            avg_decibels   = float(np.clip(avg_decibels + self._rng.uniform(20, 35), 90, 130))
            traffic_density *= float(self._rng.uniform(1.5, 2.5))
            honking_freq    *= float(self._rng.uniform(2, 4))
            is_anomaly       = True

        return {
            "timestamp":              now.isoformat(),
            "hour":                   hour,
            "day_of_week":            now.weekday(),
            "is_weekend":             is_weekend,
            "latitude":               round(lat, 6),
            "longitude":              round(lon, 6),
            "zone_type":              zone,
            "weather_code":           weather_code,
            "weather_condition":      WEATHER_MAP[weather_code],
            "temperature_c":          round(temperature_c, 2),
            "humidity_pct":           round(humidity_pct, 2),
            "wind_speed_kmh":         round(wind_speed_kmh, 2),
            "traffic_density":        round(traffic_density, 2),
            "vehicle_count":          vehicle_count,
            "honking_frequency":      round(honking_freq, 2),
            "heavy_vehicle_pct":      round(heavy_vehicle_pct, 2),
            "industrial_activity":    round(industrial_activity, 2),
            "construction_activity":  round(construction_activity, 2),
            "population_density":     round(population_density, 2),
            "event_activity":         event_activity,
            "avg_decibels":           round(avg_decibels, 2),
            "simulated_anomaly":      is_anomaly,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Alert helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_alert_level(db: float, score: float = 0.0) -> dict:
    if db >= 90 or score >= 0.80:
        return {"level": "CRITICAL", "icon": "🔴", "color": "#E63946", "action": "Immediate intervention required"}
    elif db >= 75 or score >= 0.60:
        return {"level": "HIGH",     "icon": "🟠", "color": "#F77F00", "action": "Deploy noise monitoring team"}
    elif db >= 65 or score >= 0.40:
        return {"level": "MODERATE", "icon": "🟡", "color": "#FFB703", "action": "Flag for review"}
    elif db >= 55 or score >= 0.20:
        return {"level": "LOW",      "icon": "🟢", "color": "#57CC99", "action": "Continue monitoring"}
    else:
        return {"level": "NORMAL",   "icon": "⚪", "color": "#8B949E", "action": "All clear"}


def classify_noise_severity(db: float) -> str:
    if db >= 85: return "Extreme"
    if db >= 70: return "High"
    if db >= 60: return "Moderate"
    if db >= 50: return "Low"
    return "Very Low"
