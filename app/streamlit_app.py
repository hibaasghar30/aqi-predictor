import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from datetime import timedelta
from src import config
from src.predictor import get_live_row, load_model, predict_aqi
from src.feature_store import get_feature_store
from src.config import get_aqi_category


st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="wide")


@st.cache_data(ttl=600)  # refresh at most every 10 minutes
def load_live_prediction():
    row = get_live_row()
    predictions = {}
    for horizon_name in ["24h", "48h", "72h"]:
        model, scaler, metadata = load_model(horizon_name)
        prediction = predict_aqi(row, model, scaler, metadata)
        predictions[horizon_name] = {
            "value": prediction,
            "model_used": metadata["best_model"],
        }
    return row, predictions


@st.cache_data(ttl=600)
def load_recent_history(days=7):
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features", version=1, primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi", time_travel_format="HUDI",
    )
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - timedelta(days=days)
    return df[df["timestamp"] >= cutoff]

def render_aqi_card(label, aqi_value):
    category_label, color = get_aqi_category(aqi_value)
    st.markdown(f"""
        <div style="background-color:{color}22; border-left: 6px solid {color};
                    padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;">
            <div style="font-size: 14px; color: #888;">{label}</div>
            <div style="font-size: 32px; font-weight: 700; color: {color};">{aqi_value}</div>
            <div style="font-size: 13px; color: #aaa;">{category_label}</div>
        </div>
    """, unsafe_allow_html=True)



st.title("🌫️ Karachi AQI Forecast")
st.caption("Live air quality monitoring and 3-day forecast")

with st.spinner("Fetching live data and running predictions..."):
    row, predictions = load_live_prediction()

render_aqi_card("Current AQI", row["aqi"])

st.divider()
st.subheader("Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    render_aqi_card("24 hours ahead", predictions["24h"]["value"])
    st.caption(f"Model: {predictions['24h']['model_used']}")

with col2:
    render_aqi_card("48 hours ahead", predictions["48h"]["value"])
    st.caption(f"Model: {predictions['48h']['model_used']}")

with col3:
    render_aqi_card("72 hours ahead", predictions["72h"]["value"])
    st.caption(f"Model: {predictions['72h']['model_used']}")
st.divider()
st.subheader("Current Pollutant Levels")

pollutant_col1, pollutant_col2, pollutant_col3 = st.columns(3)

with pollutant_col1:
    st.metric("PM2.5", row["pm2_5"])
    st.metric("PM10", row["pm10"])

with pollutant_col2:
    st.metric("CO", row["co"])
    st.metric("NO2", row["no2"])

with pollutant_col3:
    st.metric("SO2", row["so2"])
    st.metric("O3", row["o3"])



st.divider()
st.subheader("Recent Trend & Forecast")

history_df = load_recent_history(days=7)

chart_df = history_df[["timestamp", "aqi"]].copy()
chart_df["type"] = "Historical"

last_time = history_df["timestamp"].max()
forecast_rows = pd.DataFrame([
    {"timestamp": last_time + timedelta(hours=24), "aqi": predictions["24h"]["value"], "type": "Forecast"},
    {"timestamp": last_time + timedelta(hours=48), "aqi": predictions["48h"]["value"], "type": "Forecast"},
    {"timestamp": last_time + timedelta(hours=72), "aqi": predictions["72h"]["value"], "type": "Forecast"},
])

chart_df = pd.concat([chart_df, forecast_rows], ignore_index=True)
chart_df = chart_df.set_index("timestamp")

st.line_chart(chart_df, y="aqi", color="type")