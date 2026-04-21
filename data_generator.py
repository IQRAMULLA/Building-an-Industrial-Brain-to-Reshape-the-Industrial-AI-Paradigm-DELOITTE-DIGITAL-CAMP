"""
Industrial Brain — Synthetic Sensor Data Generator
Produces realistic multi-sensor time-series with:
  - Normal operating variation
  - Gradual degradation trends
  - Fault injection for demo / training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import MACHINES, SENSOR_RANGES, DATA_CONFIG


def _base_signal(n: int, base: float, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise around a base value."""
    return base + rng.normal(0, noise_std * base, n)


def _degradation_trend(n: int, rate: float, severity: float) -> np.ndarray:
    """Monotonically increasing degradation ramp."""
    t = np.linspace(0, 1, n)
    return severity * (np.exp(rate * t * 10) - 1)


def generate_sensor_history(
    machine_id: str,
    n_points: int = DATA_CONFIG["history_points"],
    fault_mode: str = "none",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate historical sensor readings for one machine.

    fault_mode options:
      'none'        — healthy operation
      'bearing'     — rising vibration + temperature
      'overload'    — rising current + pressure
      'wear'        — gradual across all sensors
    """
    rng = np.random.default_rng(seed)
    machine = MACHINES[machine_id]
    age_factor = 1 + machine["age_years"] * 0.03

    sr = SENSOR_RANGES
    base = {
        "vibration_rms": 2.5 * age_factor,
        "temperature_c": 55.0 * age_factor,
        "pressure_bar":  5.5,
        "rpm":           2400.0,
        "current_amp":   14.0 * age_factor,
    }

    noise = DATA_CONFIG["noise_std"]
    vib  = _base_signal(n_points, base["vibration_rms"], noise, rng)
    temp = _base_signal(n_points, base["temperature_c"], noise, rng)
    pres = _base_signal(n_points, base["pressure_bar"],  noise, rng)
    rpm  = _base_signal(n_points, base["rpm"],           noise * 0.5, rng)
    curr = _base_signal(n_points, base["current_amp"],   noise, rng)

    severity = 0.0
    if fault_mode == "bearing":
        severity = rng.uniform(0.4, 0.9)
        vib  += _degradation_trend(n_points, 0.4, severity * 6.0)
        temp += _degradation_trend(n_points, 0.2, severity * 20.0)
    elif fault_mode == "overload":
        severity = rng.uniform(0.3, 0.8)
        curr += _degradation_trend(n_points, 0.3, severity * 10.0)
        pres += _degradation_trend(n_points, 0.2, severity * 4.0)
    elif fault_mode == "wear":
        severity = rng.uniform(0.2, 0.6)
        drift = _degradation_trend(n_points, DATA_CONFIG["drift_rate"] * 100, severity)
        vib  += drift * 3
        temp += drift * 10
        curr += drift * 4

    # Clip to physical limits
    vib  = np.clip(vib,  sr["vibration_rms"]["min"], sr["vibration_rms"]["max"])
    temp = np.clip(temp, sr["temperature_c"]["min"],  sr["temperature_c"]["max"])
    pres = np.clip(pres, sr["pressure_bar"]["min"],   sr["pressure_bar"]["max"])
    rpm  = np.clip(rpm,  sr["rpm"]["min"],             sr["rpm"]["max"])
    curr = np.clip(curr, sr["current_amp"]["min"],     sr["current_amp"]["max"])

    start = datetime.now() - timedelta(seconds=n_points * DATA_CONFIG["interval_seconds"])
    timestamps = [start + timedelta(seconds=i * DATA_CONFIG["interval_seconds"])
                  for i in range(n_points)]

    df = pd.DataFrame({
        "timestamp":     timestamps,
        "machine_id":    machine_id,
        "vibration_rms": np.round(vib, 3),
        "temperature_c": np.round(temp, 2),
        "pressure_bar":  np.round(pres, 3),
        "rpm":           np.round(rpm, 1),
        "current_amp":   np.round(curr, 3),
        "fault_mode":    fault_mode,
    })
    return df


def generate_training_dataset(n_samples_per_class: int = 800) -> pd.DataFrame:
    """
    Build a labelled dataset for RUL regression training.
    Returns feature rows with RUL target (days).
    """
    frames = []
    fault_modes = ["none", "bearing", "overload", "wear"]
    seed = 100

    for machine_id in MACHINES:
        for fault in fault_modes:
            n = n_samples_per_class // len(fault_modes)
            df = generate_sensor_history(machine_id, n_points=n, fault_mode=fault, seed=seed)
            seed += 1

            # Simulate RUL: healthy → ~120 days, severe faults → near 0
            fault_penalty = {"none": 0, "bearing": 0.7, "overload": 0.5, "wear": 0.4}
            base_rul = 120 - MACHINES[machine_id]["age_years"] * 8
            noise = np.random.default_rng(seed).uniform(-5, 5, len(df))
            degradation = df["vibration_rms"] / 15 + df["temperature_c"] / 120
            rul = base_rul - fault_penalty[fault] * degradation * 60 + noise
            df["rul_days"] = np.clip(np.round(rul, 1), 0, 180)
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


def get_latest_readings(seed_offset: int = 0) -> pd.DataFrame:
    """
    Simulate one fresh reading per machine for the live dashboard.
    Each call uses a different seed to mimic streaming data.
    """
    rows = []
    fault_scenarios = {
        "CNC-A1": "none",
        "CNC-B2": "bearing",
        "PMP-C1": "none",
        "PMP-D2": "overload",
    }
    for machine_id, fault in fault_scenarios.items():
        df = generate_sensor_history(machine_id, n_points=1,
                                     fault_mode=fault,
                                     seed=seed_offset + hash(machine_id) % 1000)
        rows.append(df.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


if __name__ == "__main__":
    print("Generating training dataset...")
    df = generate_training_dataset()
    print(df.describe())
    print(f"\nTotal rows: {len(df)}")
    df.to_csv("data/training_data.csv", index=False)
    print("Saved to data/training_data.csv")
