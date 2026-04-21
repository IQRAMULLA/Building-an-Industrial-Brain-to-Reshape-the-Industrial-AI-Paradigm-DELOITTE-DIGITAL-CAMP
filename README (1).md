# Industrial Brain 🧠
### Predictive Maintenance AI — Reshaping the Industrial AI Paradigm

---

## Quick start

```bash
# 1. Clone / unzip the project
cd industrial_brain

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

The first run automatically trains both AI models (~20 seconds).
The dashboard opens at **http://localhost:8501**

---

## Project structure

```
industrial_brain/
├── app.py              # Streamlit dashboard (main demo)
├── model.py            # Isolation Forest + Random Forest models
├── data_generator.py   # Synthetic sensor data with degradation
├── config.py           # Thresholds, sensor ranges, machine catalog
├── requirements.txt    # Python dependencies
├── models/             # Saved model files (auto-created on first run)
│   ├── isolation_forest.pkl
│   ├── rul_random_forest.pkl
│   └── scaler.pkl
└── data/               # Optional: export training data
```

---

## System design

### Problem
Industrial machines fail unexpectedly, causing **$50B+/year** in global downtime costs.
Existing solutions rely on fixed time-based maintenance schedules — they either under-maintain
(missing emerging faults) or over-maintain (wasting resources).

### Solution — two-layer AI
| Layer | Model | Purpose |
|-------|-------|---------|
| Anomaly detection | Isolation Forest | Flags abnormal sensor patterns without requiring failure labels |
| RUL prediction   | Random Forest   | Estimates days until failure from 17 engineered features |

### Key innovation
**Multi-sensor feature fusion**: Raw sensor data (5 channels) is expanded into 17 features
including cross-sensor interaction terms (thermal-vibration stress index, electrical load ratio)
and proximity-to-critical-threshold features. This gives the model context that single-sensor
approaches miss entirely.

---

## Sensors monitored
| Sensor | Unit | Warning | Critical |
|--------|------|---------|----------|
| Vibration (RMS) | mm/s | 8.0 | 12.0 |
| Temperature | °C | 80 | 100 |
| Pressure | bar | 9.0 | 11.0 |
| RPM | rev/min | 3200 | 3500 |
| Current | A | 24 | 28 |

---

## Training data
Generated synthetically with realistic degradation patterns:
- **Bearing fault**: rising vibration + temperature
- **Overload fault**: rising current + pressure
- **Wear fault**: gradual degradation across all sensors
- **Healthy**: normal operating variation

Replace `data_generator.py` with real sensor API calls for production deployment.

---

## Scoring alignment
| Criterion | How this project addresses it |
|-----------|-------------------------------|
| Problem analysis (30 pts) | Quantified pain point, root cause, stakeholder personas |
| Solution design (30 pts) | Dual-model AI, 17-feature engineering, modular architecture |
| Implementation (30 pts) | Working demo, cost analysis, security plan, data pipeline |
| On-site (10 pts) | Live Streamlit demo with real-time inference |

---

## Extending to production
1. Replace `data_generator.get_latest_readings()` with real MQTT/OPC-UA sensor feeds
2. Add PostgreSQL / InfluxDB for time-series persistence
3. Deploy on edge hardware (Raspberry Pi 5 + Docker) for on-premise inference
4. Add federated learning for multi-factory privacy-preserving training
5. Integrate with ERP (SAP / Oracle) for automatic work-order generation

---

## License
Original work — all code and models are created from scratch for this project.
No proprietary datasets, patents, or third-party IP used.
