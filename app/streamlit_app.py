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


import shap
import matplotlib.pyplot as plt

#for border
import io
import base64




st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1, h2, h3 { color: #39ff14 !important; }
    hr { border-color: #39ff1444 !important; }
    </style>
""", unsafe_allow_html=True)

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
                "model_object": model,
                "feature_columns": metadata["feature_columns"],
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



@st.cache_data(ttl=600)
def compute_shap_values(_model, row, feature_columns):
    x = pd.DataFrame([row])[feature_columns]
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(x)
    return shap_values[0], x.iloc[0]


def render_shap_chart(horizon_name, model, row, feature_columns):
    shap_values, feature_values = compute_shap_values(model, row, feature_columns)

    impact = list(zip(feature_columns, shap_values))
    impact.sort(key=lambda pair: abs(pair[1]))
    top_features = impact[-6:]

    labels = [name for name, value in top_features]
    values = [value for name, value in top_features]
    colors = ["#ff4b4b" if v > 0 else "#4b8bff" for v in values]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bars = ax.barh(labels, values, color=colors, height=0.6)

    for bar, value in zip(bars, values):
        label_x = value + (0.5 if value > 0 else -0.5)
        align = "left" if value > 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                 va="center", ha=align, fontsize=9, color="white")

    ax.set_xlabel("Impact on prediction", fontsize=9, color="#aaaaaa")
    ax.set_title(f"Why the {horizon_name} prediction is what it is",
                 fontsize=11, color="white", pad=10)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.tick_params(colors="#dddddd", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    #fig.tight_layout()
    #st.markdown('<div style="border: 1px solid #ff910066; border-radius: 8px; padding: 8px; background-color: #ff91000d;">', unsafe_allow_html=True)
    #st.pyplot(fig, transparent=True)
    #st.markdown('</div>', unsafe_allow_html=True)
    #fig.tight_layout()

    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    st.markdown(f"""
        <div style="border: 1px solid #ff910066; border-radius: 8px; padding: 12px;
                    background-color: #ff91000d;">
            <img src="data:image/png;base64,{img_base64}" style="width: 100%;">
        </div>
    """, unsafe_allow_html=True)


def render_aqi_card(label, aqi_value, glow_color=None, number_color=None):
    category_label, severity_color = get_aqi_category(aqi_value)
    box_color = glow_color if glow_color else severity_color
    text_color = number_color if number_color else severity_color
    st.markdown(f"""
        <div style="background-color:{box_color}22; border: 1px solid {box_color}88;
                    border-left: 6px solid {severity_color}; padding: 12px 16px;
                    border-radius: 8px; margin-bottom: 8px;
                    box-shadow: 0 0 12px {box_color}55;">
            <div style="font-size: 14px; color: #888;">{label}</div>
            <div style="font-size: 32px; font-weight: 700; color: {text_color};">{aqi_value}</div>
            <div style="font-size: 13px; color: #aaa;">{category_label}</div>
        </div>
    """, unsafe_allow_html=True)

st.title("🌫️ Karachi AQI Forecast")
st.caption("Live air quality monitoring and 3-day forecast")

with st.spinner("Fetching live data and running predictions..."):
    row, predictions = load_live_prediction()

render_aqi_card("Current AQI", row["aqi"], glow_color="#b026ff",number_color="#ff10f0")

st.divider()
st.subheader("Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    render_aqi_card("24 hours ahead", predictions["24h"]["value"], glow_color="#b026ff",number_color="#ff10f0")
    st.caption(f"Model: {predictions['24h']['model_used']}")
    render_shap_chart("24h", predictions["24h"]["model_object"], row, predictions["24h"]["feature_columns"])
with col2:
    render_aqi_card("48 hours ahead", predictions["48h"]["value"], glow_color="#b026ff",number_color="#ff10f0")
    st.caption(f"Model: {predictions['48h']['model_used']}")
    render_shap_chart("48h", predictions["48h"]["model_object"], row, predictions["48h"]["feature_columns"])

with col3:
    render_aqi_card("72 hours ahead", predictions["72h"]["value"], glow_color="#b026ff",number_color="#ff10f0")
    st.caption(f"Model: {predictions['72h']['model_used']}")
    render_shap_chart("72h", predictions["72h"]["model_object"], row, predictions["72h"]["feature_columns"])


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