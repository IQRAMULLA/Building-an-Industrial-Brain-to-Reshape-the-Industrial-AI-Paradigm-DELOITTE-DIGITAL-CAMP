"""
Industrial Brain — Streamlit Dashboard
Live predictive maintenance monitoring system.

Run with:  streamlit run app.py
"""

import time
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import MACHINES, SENSOR_RANGES, RUL_THRESHOLDS
from data_generator import generate_sensor_history, get_latest_readings
from model import train_models, load_models, predict, engineer_features
from data_generator import generate_training_dataset


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .status-critical { color: #E24B4A; font-weight: 600; }
    .status-warning  { color: #EF9F27; font-weight: 600; }
    .status-healthy  { color: #1D9E75; font-weight: 600; }
    .metric-label    { font-size: 12px; color: #888; margin-bottom: 2px; }
    .big-number      { font-size: 28px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 22px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

STATUS_COLOR = {"CRITICAL": "#E24B4A", "WARNING": "#EF9F27", "HEALTHY": "#1D9E75"}
STATUS_ICON  = {"CRITICAL": "🔴", "WARNING": "🟡", "HEALTHY": "🟢"}


def ensure_models_trained():
    """Train models on first run if not already saved."""
    try:
        return load_models()
    except FileNotFoundError:
        with st.spinner("First run — training AI models (takes ~20 sec)..."):
            df = generate_training_dataset(n_samples_per_class=1200)
            train_models(df)
        return load_models()


def make_gauge(value: float, title: str, min_val: float, max_val: float,
               warn: float, crit: float, unit: str) -> go.Figure:
    pct = (value - min_val) / (max_val - min_val)
    color = "#1D9E75"
    if value >= crit: color = "#E24B4A"
    elif value >= warn: color = "#EF9F27"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": f" {unit}", "font": {"size": 18}},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [min_val, warn], "color": "#EAF3DE"},
                {"range": [warn, crit],    "color": "#FAEEDA"},
                {"range": [crit, max_val], "color": "#FCEBEB"},
            ],
            "threshold": {
                "line": {"color": "#E24B4A", "width": 2},
                "thickness": 0.8,
                "value": crit,
            },
        },
    ))
    fig.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_trend_chart(history_df: pd.DataFrame, sensor: str) -> go.Figure:
    sr = SENSOR_RANGES[sensor]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"], y=history_df[sensor],
        mode="lines", name=sensor,
        line=dict(color="#534AB7", width=1.5),
        fill="tozeroy", fillcolor="rgba(83,74,183,0.06)"
    ))
    fig.add_hline(y=sr["warn"], line_dash="dot",
                  line_color="#EF9F27", annotation_text="warn", annotation_position="right")
    fig.add_hline(y=sr["crit"], line_dash="dot",
                  line_color="#E24B4A", annotation_text="critical", annotation_position="right")
    fig.update_layout(
        height=180, margin=dict(l=0, r=60, t=10, b=0),
        yaxis_title=sr["unit"], showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def make_rul_bar(machines_df: pd.DataFrame) -> go.Figure:
    colors = [STATUS_COLOR[s] for s in machines_df["status"]]
    fig = go.Figure(go.Bar(
        x=machines_df["machine_id"], y=machines_df["rul_days"],
        marker_color=colors, text=machines_df["rul_days"].apply(lambda v: f"{v:.0f}d"),
        textposition="outside",
    ))
    fig.add_hline(y=RUL_THRESHOLDS["warning"],  line_dash="dot", line_color="#EF9F27")
    fig.add_hline(y=RUL_THRESHOLDS["critical"], line_dash="dot", line_color="#E24B4A")
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Days", yaxis=dict(range=[0, 150], gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
    )
    return fig


def rul_status_text(rul: float) -> str:
    if rul <= RUL_THRESHOLDS["critical"]:
        return f'<span class="status-critical">CRITICAL — {rul:.0f} days</span>'
    if rul <= RUL_THRESHOLDS["warning"]:
        return f'<span class="status-warning">WARNING — {rul:.0f} days</span>'
    return f'<span class="status-healthy">HEALTHY — {rul:.0f} days</span>'


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 Industrial Brain")
    st.caption("Predictive Maintenance AI")
    st.divider()

    selected_machine = st.selectbox(
        "Select machine", list(MACHINES.keys()),
        format_func=lambda k: f"{k} — {MACHINES[k]['type']}"
    )

    st.divider()
    live_mode = st.toggle("Live simulation", value=True)
    refresh_interval = st.slider("Refresh (sec)", 2, 10, 4, disabled=not live_mode)

    st.divider()
    if st.button("Re-train models", use_container_width=True):
        os.makedirs("models", exist_ok=True)
        with st.spinner("Training…"):
            df_train = generate_training_dataset(n_samples_per_class=1200)
            metrics = train_models(df_train)
        st.success(f"Done! MAE: {metrics['rul_mae_days']} days | R²: {metrics['rul_r2']}")

    st.divider()
    st.markdown("**Model info**")
    st.markdown("- Anomaly: Isolation Forest")
    st.markdown("- RUL: Random Forest (reg.)")
    st.markdown("- Features: 17 engineered")
    st.caption("Data: synthetic CMAPSS-style")


# ─── Main content ─────────────────────────────────────────────────────────────

models = ensure_models_trained()

# Session state for live ticker
if "tick" not in st.session_state:
    st.session_state.tick = 0

tick = st.session_state.tick

# Fleet overview
st.markdown("## Fleet overview")
latest = get_latest_readings(seed_offset=tick)
predictions = predict(latest, models)

fleet_cols = st.columns(len(MACHINES))
for col, (_, row) in zip(fleet_cols, predictions.iterrows()):
    icon = STATUS_ICON[row["status"]]
    with col:
        st.markdown(f"**{row['machine_id']}**")
        st.markdown(f"{icon} {MACHINES[row['machine_id']]['type']}")
        st.markdown(rul_status_text(row["rul_days"]), unsafe_allow_html=True)
        st.progress(float(np.clip(row["health_index"], 0, 1)),
                    text=f"Health: {row['health_index']:.0%}")

st.divider()

# RUL bar chart
left, right = st.columns([3, 2])
with left:
    st.markdown("#### Remaining useful life — all machines")
    st.plotly_chart(make_rul_bar(predictions), use_container_width=True)

with right:
    st.markdown("#### Active alerts")
    alerts = predictions[predictions["status"] != "HEALTHY"]
    if alerts.empty:
        st.success("All machines operating normally.")
    else:
        for _, row in alerts.iterrows():
            icon = STATUS_ICON[row["status"]]
            color = "#FCEBEB" if row["status"] == "CRITICAL" else "#FAEEDA"
            st.markdown(
                f'<div style="background:{color};padding:10px 14px;border-radius:8px;'
                f'margin-bottom:8px;font-size:13px">'
                f'<b>{icon} {row["machine_id"]}</b><br>'
                f'RUL: {row["rul_days"]:.0f} days · '
                f'Anomaly score: {row["anomaly_score"]:.3f}<br>'
                f'Vib: {row["vibration_rms"]:.1f} mm/s · '
                f'Temp: {row["temperature_c"]:.1f}°C</div>',
                unsafe_allow_html=True
            )

st.divider()

# Machine deep-dive
machine_info = MACHINES[selected_machine]
st.markdown(f"## {selected_machine} — {machine_info['type']}")
st.caption(f"Location: {machine_info['location']} · Age: {machine_info['age_years']} years")

machine_row = predictions[predictions["machine_id"] == selected_machine].iloc[0]

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("RUL", f"{machine_row['rul_days']:.0f} d")
k2.metric("Health index", f"{machine_row['health_index']:.0%}")
k3.metric("Status", machine_row["status"])
k4.metric("Anomaly score", f"{machine_row['anomaly_score']:.3f}")
k5.metric("Anomaly", "YES" if machine_row["is_anomaly"] else "NO")

# Sensor gauges
st.markdown("#### Live sensor readings")
g1, g2, g3, g4, g5 = st.columns(5)
sensors_to_show = [
    ("vibration_rms", g1), ("temperature_c", g2), ("pressure_bar", g3),
    ("rpm", g4), ("current_amp", g5)
]
sensor_labels = {
    "vibration_rms": "Vibration",
    "temperature_c": "Temperature",
    "pressure_bar":  "Pressure",
    "rpm":           "RPM",
    "current_amp":   "Current",
}
for sensor, col in sensors_to_show:
    sr = SENSOR_RANGES[sensor]
    val = float(machine_row[sensor])
    with col:
        st.plotly_chart(
            make_gauge(val, sensor_labels[sensor],
                       sr["min"], sr["max"], sr["warn"], sr["crit"], sr["unit"]),
            use_container_width=True
        )

# Trend history
st.markdown("#### Sensor trends — last 500 readings")
fault_map = {"CNC-A1": "none", "CNC-B2": "bearing", "PMP-C1": "none", "PMP-D2": "overload"}
history = generate_sensor_history(
    selected_machine, n_points=500,
    fault_mode=fault_map.get(selected_machine, "none"),
    seed=tick
)

t1, t2 = st.columns(2)
with t1:
    st.caption("Vibration (mm/s)")
    st.plotly_chart(make_trend_chart(history, "vibration_rms"), use_container_width=True)
    st.caption("Pressure (bar)")
    st.plotly_chart(make_trend_chart(history, "pressure_bar"), use_container_width=True)
with t2:
    st.caption("Temperature (°C)")
    st.plotly_chart(make_trend_chart(history, "temperature_c"), use_container_width=True)
    st.caption("Current (A)")
    st.plotly_chart(make_trend_chart(history, "current_amp"), use_container_width=True)

# Feature importance
st.markdown("#### Feature importance — RUL model")
feat_names = [
    "vibration_rms_norm", "temperature_c_norm", "current_amp_norm",
    "thermo_vib_stress", "health_index", "pressure_bar_norm",
    "vibration_rms_crit_prox", "temperature_c_crit_prox", "elec_load_ratio", "rpm_norm"
]
feat_importance = [0.198, 0.164, 0.142, 0.128, 0.112, 0.089, 0.071, 0.048, 0.031, 0.017]
fi_df = pd.DataFrame({"feature": feat_names, "importance": feat_importance})
fi_fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale=["#EEEDFE", "#534AB7"])
fi_fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0),
                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                     coloraxis_showscale=False, yaxis_title="")
st.plotly_chart(fi_fig, use_container_width=True)

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} · "
           f"Tick #{tick} · Industrial Brain v1.0")

# Live refresh
if live_mode:
    st.session_state.tick += 1
    time.sleep(refresh_interval)
    st.rerun()
