"""
Industrial Brain — AI Model Layer
Two-model approach:
  1. IsolationForest  — unsupervised anomaly detection (no labels needed)
  2. RandomForest     — supervised RUL (days to failure) regression
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from config import MODEL_PARAMS, SENSOR_RANGES, RUL_THRESHOLDS

MODEL_DIR = "models"
ANOMALY_MODEL_PATH  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
RUL_MODEL_PATH      = os.path.join(MODEL_DIR, "rul_random_forest.pkl")
SCALER_PATH         = os.path.join(MODEL_DIR, "scaler.pkl")

RAW_FEATURES = [
    "vibration_rms", "temperature_c", "pressure_bar", "rpm", "current_amp"
]


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand raw sensor readings into a richer feature vector.
    Adds normalised values, interaction terms, and statistical moments
    computed over a rolling window (when sufficient history is available).
    """
    feat = df[RAW_FEATURES].copy()

    # Normalise each sensor to [0, 1] based on physical range
    for col in RAW_FEATURES:
        lo = SENSOR_RANGES[col]["min"]
        hi = SENSOR_RANGES[col]["max"]
        feat[f"{col}_norm"] = (df[col] - lo) / (hi - lo)

    # Proximity to critical threshold (0 = healthy, 1 = at critical limit)
    for col in RAW_FEATURES:
        crit = SENSOR_RANGES[col]["crit"]
        hi   = SENSOR_RANGES[col]["max"]
        lo   = SENSOR_RANGES[col]["min"]
        feat[f"{col}_crit_prox"] = np.clip(
            (df[col] - lo) / (crit - lo), 0, 1
        )

    # Interaction: thermal-vibration stress index
    feat["thermo_vib_stress"] = feat["vibration_rms_norm"] * feat["temperature_c_norm"]

    # Electrical load ratio
    feat["elec_load_ratio"] = feat["current_amp_norm"] / (feat["rpm_norm"] + 1e-6)

    # Composite health index (lower = less healthy)
    feat["health_index"] = 1.0 - (
        0.35 * feat["vibration_rms_norm"] +
        0.25 * feat["temperature_c_norm"] +
        0.20 * feat["current_amp_norm"] +
        0.10 * feat["pressure_bar_norm"] +
        0.10 * feat["rpm_norm"]
    )

    return feat


def compute_window_stats(series: pd.Series) -> dict:
    """Statistical features over a rolling window (used for live inference)."""
    arr = series.values
    if len(arr) < 2:
        return {"mean": arr[-1], "std": 0, "kurtosis": 0, "skew": 0, "trend": 0}
    return {
        "mean":     float(np.mean(arr)),
        "std":      float(np.std(arr)),
        "kurtosis": float(kurtosis(arr)),
        "skew":     float(skew(arr)),
        "trend":    float(np.polyfit(np.arange(len(arr)), arr, 1)[0]),
    }


# ─── Model Training ───────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    """
    Train both models on the provided dataframe.
    Saves models to disk and returns evaluation metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    feat_df = engineer_features(df)
    feature_cols = [c for c in feat_df.columns]

    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df[feature_cols])
    joblib.dump(scaler, SCALER_PATH)

    # ── 1. Isolation Forest (anomaly detection) ──────────────────────────────
    iso_params = MODEL_PARAMS["isolation_forest"]
    iso = IsolationForest(**iso_params)
    iso.fit(X_scaled)
    joblib.dump(iso, ANOMALY_MODEL_PATH)
    anomaly_scores = iso.decision_function(X_scaled)

    # ── 2. Random Forest RUL Regression ──────────────────────────────────────
    if "rul_days" not in df.columns:
        raise ValueError("Training data must contain 'rul_days' column.")

    y = df["rul_days"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    rf_params = MODEL_PARAMS["random_forest_rul"]
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    joblib.dump(rf, RUL_MODEL_PATH)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    feature_importances = dict(zip(feature_cols, rf.feature_importances_))

    print(f"[RUL Model] MAE: {mae:.1f} days | R²: {r2:.3f}")
    return {
        "rul_mae_days": round(mae, 1),
        "rul_r2":       round(r2, 3),
        "feature_importances": feature_importances,
        "anomaly_contamination": iso_params["contamination"],
    }


# ─── Inference ────────────────────────────────────────────────────────────────

def load_models():
    """Load trained models from disk."""
    if not os.path.exists(RUL_MODEL_PATH):
        raise FileNotFoundError(
            "Models not found. Run: python model.py  (or click 'Train' in the dashboard)"
        )
    return {
        "iso":    joblib.load(ANOMALY_MODEL_PATH),
        "rf":     joblib.load(RUL_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
    }


def predict(readings_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run inference on a batch of sensor readings.
    Returns enriched dataframe with anomaly flag, anomaly score,
    predicted RUL, and health status.
    """
    feat_df   = engineer_features(readings_df)
    feat_cols = list(feat_df.columns)
    X_scaled  = models["scaler"].transform(feat_df[feat_cols])

    anomaly_scores = models["iso"].decision_function(X_scaled)
    is_anomaly     = models["iso"].predict(X_scaled) == -1   # -1 = anomaly

    rul_pred = models["rf"].predict(X_scaled)
    rul_pred = np.clip(np.round(rul_pred, 1), 0, 180)

    result = readings_df.copy()
    result["anomaly_score"]  = np.round(anomaly_scores, 4)
    result["is_anomaly"]     = is_anomaly
    result["rul_days"]       = rul_pred
    result["health_index"]   = np.round(feat_df["health_index"].values, 3)

    def status(row):
        if row["is_anomaly"] or row["rul_days"] <= RUL_THRESHOLDS["critical"]:
            return "CRITICAL"
        if row["rul_days"] <= RUL_THRESHOLDS["warning"]:
            return "WARNING"
        return "HEALTHY"

    result["status"] = result.apply(status, axis=1)
    return result


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_generator import generate_training_dataset
    print("Generating training data...")
    df = generate_training_dataset(n_samples_per_class=1200)
    print(f"Training on {len(df)} samples...")
    metrics = train_models(df)
    print("\n=== Training complete ===")
    for k, v in metrics.items():
        if k != "feature_importances":
            print(f"  {k}: {v}")
    print("\nTop 5 features by importance:")
    fi = metrics["feature_importances"]
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.4f}")
