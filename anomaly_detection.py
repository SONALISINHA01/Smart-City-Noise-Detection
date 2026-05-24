"""
================================================================================
  Anomaly Detection Module
  Isolation Forest · Autoencoder (TensorFlow/Keras)
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestDetector:
    """Sklearn Isolation Forest wrapper with anomaly-score utilities."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples="auto",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = normal, -1 = anomaly."""
        return self.model.predict(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Negate the raw decision-function so that HIGHER score = MORE anomalous.
        Normalise to [0, 1].
        """
        raw    = -self.model.decision_function(X)
        normed = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return normed

    def save(self, path: str = "models/isolation_forest.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  ✅ Isolation Forest saved → {path}")

    @classmethod
    def load(cls, path: str = "models/isolation_forest.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Autoencoder (TensorFlow/Keras)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseAutoencoder:
    """
    Deep Autoencoder for unsupervised anomaly detection.
    High reconstruction error  →  anomalous sample.
    """

    def __init__(
        self,
        input_dim:      int   = 23,
        encoding_dim:   int   = 8,
        latent_dim:     int   = 4,
        dropout_rate:   float = 0.2,
        learning_rate:  float = 1e-3,
    ):
        self.input_dim     = input_dim
        self.encoding_dim  = encoding_dim
        self.latent_dim    = latent_dim
        self.dropout_rate  = dropout_rate
        self.learning_rate = learning_rate
        self.threshold_    = None
        self.history_      = None
        self._build()

    def _build(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers
        except ImportError:
            raise ImportError("TensorFlow is required for Autoencoder. pip install tensorflow")

        inp = keras.Input(shape=(self.input_dim,), name="input")

        # Encoder
        x = layers.Dense(64, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.encoding_dim, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Dense(self.latent_dim, activation="relu", name="latent")(x)

        # Decoder
        x = layers.Dense(self.encoding_dim, activation="relu")(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        decoded = layers.Dense(self.input_dim, activation="linear", name="output")(x)

        self.autoencoder = keras.Model(inp, decoded, name="NoiseAutoencoder")
        self.encoder     = keras.Model(inp, encoded, name="Encoder")

        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"]
        )

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 60,
        batch_size: int = 64,
        validation_split: float = 0.15,
        contamination_pct: float = 95.0,
        verbose: int = 1,
    ):
        try:
            import tensorflow as tf
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
                ),
            ]
        except ImportError:
            callbacks = []

        self.history_ = self.autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Compute reconstruction errors and set threshold at `contamination_pct` percentile
        recon_errors = self._reconstruction_error(X)
        self.threshold_ = np.percentile(recon_errors, contamination_pct)
        print(f"  ✅ Autoencoder trained | threshold={self.threshold_:.6f}")
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.autoencoder.predict(X, verbose=0)
        return np.mean((X - X_hat) ** 2, axis=1)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Normalised [0,1] reconstruction error."""
        errors = self._reconstruction_error(X)
        normed = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
        return normed

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 = anomaly, 0 = normal based on threshold."""
        errors = self._reconstruction_error(X)
        return (errors > self.threshold_).astype(int)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.predict(X, verbose=0)

    def save(self, dir_path: str = "models/autoencoder"):
        os.makedirs(dir_path, exist_ok=True)
        self.autoencoder.save(os.path.join(dir_path, "autoencoder.keras"))
        self.encoder.save(os.path.join(dir_path, "encoder.keras"))
        meta = {"threshold": self.threshold_, "input_dim": self.input_dim}
        joblib.dump(meta, os.path.join(dir_path, "meta.pkl"))
        print(f"  ✅ Autoencoder saved → {dir_path}/")

    @classmethod
    def load(cls, dir_path: str = "models/autoencoder"):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required")
        meta = joblib.load(os.path.join(dir_path, "meta.pkl"))
        obj  = cls.__new__(cls)
        obj.autoencoder = tf.keras.models.load_model(os.path.join(dir_path, "autoencoder.keras"))
        obj.encoder     = tf.keras.models.load_model(os.path.join(dir_path, "encoder.keras"))
        obj.threshold_  = meta["threshold"]
        obj.input_dim   = meta["input_dim"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Ensemble anomaly score
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_anomaly_score(
    if_scores: np.ndarray,
    ae_scores: np.ndarray,
    w_if: float = 0.5,
    w_ae: float = 0.5,
) -> np.ndarray:
    """Weighted average of Isolation Forest + Autoencoder anomaly scores."""
    return w_if * if_scores + w_ae * ae_scores


def get_alert_level(score: float) -> str:
    if   score >= 0.80: return "🔴 CRITICAL"
    elif score >= 0.60: return "🟠 HIGH"
    elif score >= 0.40: return "🟡 MODERATE"
    elif score >= 0.20: return "🟢 LOW"
    else:               return "⚪ NORMAL"
