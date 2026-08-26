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

from src.predictor import get_live_row, load_model, predict_aqi, check_hazard_alerts, log_hazard_alerts


#for the page hting
if "view" not in st.session_state:
    st.session_state.view = "main"


st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="wide")
#st.markdown("""
 #   <style>
  #  .stApp { background-color: #000000; }
   # h1, h2, h3 { color: #39ff14 !important; }
    #hr { border-color: #39ff1444 !important; }
    #</style>
#""", unsafe_allow_html=True)


st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1, h2, h3 { color: #39ff14 !important; }
    hr { border-color: #39ff1444 !important; }

    @media (max-width: 640px) {
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-bottom: 16px;
        }
    }
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


@st.cache_data(ttl=3600)
def load_yearly_trend():
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features", version=1, primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi", time_travel_format="HUDI",
    )
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp")

    df["year"] = df["timestamp"].dt.year
    df["month_sort"] = df["timestamp"].dt.to_period("M")
    df["month"] = df["timestamp"].dt.strftime("%b")

    monthly_avg = df.groupby(["year", "month_sort", "month"])["aqi"].mean().reset_index()
    monthly_avg = monthly_avg.sort_values("month_sort")
    monthly_avg = monthly_avg[["year", "month", "aqi"]]
    monthly_avg.columns = ["year", "month", "avg_aqi"]

    return monthly_avg


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

    value_range = max(values) - min(values)
    offset = value_range * 0.08

    # widen the x-axis so offset text has room and doesn't collide with tick labels
    ax.set_xlim(min(values) - value_range * 0.25, max(values) + value_range * 0.25)

    for bar, value in zip(bars, values):
        label_x = value + (offset if value > 0 else -offset)
        align = "left" if value > 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                 va="center", ha=align, fontsize=9, color="white")

    ax.set_xlabel("Impact on prediction", fontsize=9, color="#aaaaaa")
    ax.set_title(f"Why the {horizon_name} prediction is what it is",
                 fontsize=11, color="white", pad=10)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.tick_params(colors="#dddddd", labelsize=9, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.28)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    #st.markdown(f"""
     #   <div style="border: 1px solid #b026ff66; border-radius: 8px; padding: 12px;
      #              background-color: #b026ff0d;">

    st.markdown(f"""
        <div style="border: 1px solid #b026ff66; border-radius: 8px; padding: 12px;
                    background-color: #b026ff0d; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_base64}" style="width: 100%;">
        </div>
    """, unsafe_allow_html=True)


def render_pollutant_card(label, value):
    st.markdown(f"""
        <div style="border: 1px solid #b026ff66; border-radius: 8px; padding: 12px 16px;
                    background-color: #b026ff0d; margin-bottom: 20px;
                    box-shadow: 0 0 8px #b026ff33;">
            <div style="font-size: 13px; color: #888;">{label}</div>
            <div style="font-size: 24px; font-weight: 600; color: white;">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_aqi_card(label, aqi_value, glow_color=None, number_color=None):
    category_label, severity_color = get_aqi_category(aqi_value)
    box_color = glow_color if glow_color else severity_color
    text_color = number_color if number_color else severity_color
    st.markdown(f"""
        <div style="background-color:{box_color}22; border: 1px solid {box_color}88;
                    border-left: 6px solid {severity_color}; padding: 12px 16px;
                    border-radius: 8px; margin-bottom: 20px;min-height: 170px;;
                    box-shadow: 0 0 12px {box_color}55;">
            <div style="font-size: 14px; color: #888;">{label}</div>
            <div style="font-size: 32px; font-weight: 700; color: {text_color};">{aqi_value}</div>
            <div style="font-size: 13px; color: #aaa;">{category_label}</div>
        </div>
    """, unsafe_allow_html=True)



def get_health_advice(category_label):
    advice_map = {
        "Good, Have a fun day outside": "Air quality is good — safe for all outdoor activities.",
        "Moderate, mostly fine to go outside": "Acceptable for most people. Sensitive groups (asthma, elderly, children) should limit prolonged outdoor exertion.",
        "Unhealthy for sensitive groups, be a little careful": "Sensitive groups should reduce outdoor activity. Consider a mask if you have respiratory issues.",
        "Unhealthy, better to not make plans": "Everyone should limit outdoor exertion. Wear a mask (N95) if going outside.",
        "Very unhealthy, stay indoors if you can": "Avoid outdoor activity. Keep windows closed. Use an air purifier indoors if available.",
        "Hazardous, avoid going outside": "Stay indoors. Avoid all outdoor exertion. Keep windows closed and use an air purifier if available.",
    }
    return advice_map.get(category_label, "No specific guidance available for this air quality level.")


def render_health_card(aqi_value):
    category_label, severity_color = get_aqi_category(aqi_value)
    advice = get_health_advice(category_label)
    st.markdown(f"""
        <div style="border: 1px solid {severity_color}88; border-left: 6px solid {severity_color};
                    border-radius: 8px; padding: 12px 16px; background-color: {severity_color}11;
                    margin-bottom: 20px; min-height: 170px;">
            <div style="font-size: 13px; color: #888;">Health Guidance</div>
            <div style="font-size: 15px; color: white; margin-top: 4px;">{advice}</div>
        </div>
    """, unsafe_allow_html=True)


def render_best_time_card(current_aqi, predictions):
    options = [
        ("Now", current_aqi),
        ("In 24 hours", predictions["24h"]["value"]),
        ("In 48 hours", predictions["48h"]["value"]),
        ("In 72 hours", predictions["72h"]["value"]),
    ]

    best_label, best_value = min(options, key=lambda pair: pair[1])
    category_label, severity_color = get_aqi_category(best_value)

    st.markdown(f"""
        <div style="border: 1px solid {severity_color}88; border-left: 6px solid {severity_color};
                    border-radius: 8px; padding: 12px 16px; background-color: {severity_color}11;
                    margin-bottom: 20px; min-height: 170px;">
            <div style="font-size: 13px; color: #888;">Best Time for Outdoor Activity</div>
            <div style="font-size: 18px; color: white; margin-top: 4px; font-weight: 600;">{best_label}</div>
            <div style="font-size: 13px; color: #aaa;">Predicted AQI: {best_value:.1f} ({category_label.split(',')[0]})</div>
        </div>
    """, unsafe_allow_html=True)

def render_weather_metric_card(label, value):
    st.markdown(f"""
        <div style="background-color:#b026ff22; border: 1px solid #b026ff88;
                    border-left: 6px solid #b026ff; padding: 12px 16px;
                    border-radius: 8px; margin-bottom: 8px;
                    box-shadow: 0 0 12px #b026ff55;">
            <div style="font-size: 14px; color: #888;">{label}</div>
            <div style="font-size: 32px; font-weight: 700; color: #ff10f0;">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def render_weather_card(row):
    st.subheader("Current Weather")
    weather_col1, weather_col2, weather_col3 = st.columns(3)

    with weather_col1:
        render_weather_metric_card("Temperature", f"{row['temperature']}°C")
    with weather_col2:
        render_weather_metric_card("Humidity", f"{row['humidity']}%")
    with weather_col3:
        render_weather_metric_card("Wind Speed", f"{row['wind_speed']} m/s")



def render_monthly_bar_chart(data, title):
    colors = [get_aqi_category(v)[1] for v in data["avg_aqi"]]

    fig, ax = plt.subplots(figsize=(8, 3))
    plt.style.use("dark_background")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.bar(data["month"], data["avg_aqi"], color=colors)

    ax.set_ylabel("Average AQI", color="#aaaaaa")
    ax.set_title(title, color="white", fontsize=13)
    ax.tick_params(colors="#dddddd", labelsize=8, rotation=45)
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    st.markdown(f"""
    <div style="border: 1px solid #b026ff66; border-radius: 8px; padding: 12px;
                background-color: #b026ff0d; max-width: 600px; margin: 0 auto;">
        <img src="data:image/png;base64,{img_base64}" style="width: 100%;">
    </div>
""", unsafe_allow_html=True)




#adds the pollutant bar chart 
def render_pollutant_comparison_chart(row):
    pollutants = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]
    labels = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
    values = [row[p] for p in pollutants]

    bar_colors = ["#ff10f0", "#b026ff", "#00e4ff", "#39ff14", "#ffff00", "#ff7e00"]

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use("dark_background")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.bar(labels, values, color=bar_colors)

    ax.set_ylabel("Concentration", color="#aaaaaa")
    ax.set_title("Current Pollutant Comparison", color="white", fontsize=13)
    ax.tick_params(colors="#dddddd", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    st.markdown(f"""
    <div style="border: 1px solid #b026ff66; border-radius: 8px; padding: 12px;
                background-color: #b026ff0d; max-width: 600px; margin: 0 auto;">
        <img src="data:image/png;base64,{img_base64}" style="width: 100%;">
    </div>
""", unsafe_allow_html=True)




if st.session_state.view == "main":
    st.title("🌫️ Karachi AQI Forecast")

#adds the button of pollutants
    button_col1, button_col2, button_spacer = st.columns([1, 1, 4])
    with button_col1:
        if st.button("📅 Show AQI Monthly Trend"):
            st.session_state.view = "yearly_chart"
            st.rerun()
    with button_col2:
        if st.button("🧪 Show Pollutant Comparison"):
            st.session_state.view = "pollutant_chart"
            st.rerun()
    st.caption("Live air quality monitoring and 3-day forecast")

    with st.spinner("Fetching live data and running predictions..."):
        row, predictions = load_live_prediction()

    alerts = check_hazard_alerts(row, predictions)
    if alerts:
        st.error("⚠️ **Hazard Alert**\n\n" + "\n\n".join(alerts))
        log_hazard_alerts(alerts)

    


    aqi_col1, aqi_col2, aqi_col3 = st.columns(3)

    with aqi_col1:
        render_aqi_card("Current AQI", row["aqi"], glow_color="#b026ff", number_color="#ff10f0")
    with aqi_col2:
        render_health_card(row["aqi"])
    with aqi_col3:
        render_best_time_card(row["aqi"], predictions)


    render_weather_card(row)

    st.divider()
    st.subheader("Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_aqi_card("24 hours ahead", predictions["24h"]["value"], glow_color="#b026ff", number_color="#ff10f0")
        st.caption(f"Model: {predictions['24h']['model_used']}")
        render_shap_chart("24h", predictions["24h"]["model_object"], row, predictions["24h"]["feature_columns"])
    with col2:
        render_aqi_card("48 hours ahead", predictions["48h"]["value"], glow_color="#b026ff", number_color="#ff10f0")
        st.caption(f"Model: {predictions['48h']['model_used']}")
        render_shap_chart("48h", predictions["48h"]["model_object"], row, predictions["48h"]["feature_columns"])
    with col3:
        render_aqi_card("72 hours ahead", predictions["72h"]["value"], glow_color="#b026ff", number_color="#ff10f0")
        st.caption(f"Model: {predictions['72h']['model_used']}")
        render_shap_chart("72h", predictions["72h"]["model_object"], row, predictions["72h"]["feature_columns"])

    st.divider()
    st.subheader("Current Pollutant Levels")

    pollutant_col1, pollutant_col2, pollutant_col3 = st.columns(3)

    with pollutant_col1:
        render_pollutant_card("PM2.5", row["pm2_5"])
        render_pollutant_card("PM10", row["pm10"])

    with pollutant_col2:
        render_pollutant_card("CO", row["co"])
        render_pollutant_card("NO2", row["no2"])

    with pollutant_col3:
        render_pollutant_card("SO2", row["so2"])
        render_pollutant_card("O3", row["o3"])

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


elif st.session_state.view == "yearly_chart":
    if st.button("⬅ Back"):
        st.session_state.view = "main"
        st.rerun()

    monthly_avg = load_yearly_trend()

    available_years = sorted(monthly_avg["year"].unique(), reverse=True)
    selected_year = st.selectbox("Select year", available_years)

    year_data = monthly_avg[monthly_avg["year"] == selected_year]
    render_monthly_bar_chart(year_data, f"AQI Monthly Trend — {selected_year}")

elif st.session_state.view == "pollutant_chart":
    if st.button("⬅ Back"):
        st.session_state.view = "main"
        st.rerun()

    with st.spinner("Fetching live data..."):
        row, predictions = load_live_prediction()

    render_pollutant_comparison_chart(row)