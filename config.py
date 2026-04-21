"""
Industrial Brain — Configuration
Central place for all thresholds, sensor ranges, and model parameters.
"""

MACHINES = {
    "CNC-A1": {"type": "CNC Mill",       "location": "Zone A", "age_years": 4},
    "CNC-B2": {"type": "CNC Lathe",      "location": "Zone B", "age_years": 7},
    "PMP-C1": {"type": "Hydraulic Pump", "location": "Zone C", "age_years": 2},
    "PMP-D2": {"type": "Coolant Pump",   "location": "Zone D", "age_years": 9},
}

SENSOR_RANGES = {
    "vibration_rms":  {"min": 0.1,  "max": 15.0,  "unit": "mm/s",  "warn": 8.0,  "crit": 12.0},
    "temperature_c":  {"min": 20.0, "max": 120.0, "unit": "°C",    "warn": 80.0, "crit": 100.0},
    "pressure_bar":   {"min": 0.5,  "max": 12.0,  "unit": "bar",   "warn": 9.0,  "crit": 11.0},
    "rpm":            {"min": 100,  "max": 3600,  "unit": "RPM",   "warn": 3200, "crit": 3500},
    "current_amp":    {"min": 1.0,  "max": 30.0,  "unit": "A",     "warn": 24.0, "crit": 28.0},
}

MODEL_PARAMS = {
    "isolation_forest": {
        "n_estimators": 200,
        "contamination": 0.05,
        "random_state": 42,
    },
    "random_forest_rul": {
        "n_estimators": 300,
        "max_depth": 12,
        "random_state": 42,
    },
}

# Remaining Useful Life thresholds (days)
RUL_THRESHOLDS = {
    "critical": 7,
    "warning":  30,
    "healthy":  90,
}

DATA_CONFIG = {
    "history_points":  500,   # sensor readings per machine in history
    "interval_seconds": 5,    # simulated reading interval
    "noise_std":        0.08,  # base sensor noise
    "drift_rate":       0.002, # gradual degradation rate
}
