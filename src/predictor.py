import joblib           #saves trained model file
import json
import pandas as pd
from src import config
from src.openweather_client import geocode, get_current_weather, get_air_pollution
from src.feature_engineering import build_feature_row
import hopsworks
from pathlib import Path
from src.feature_store import get_feature_store
from src.feature_store import save_row

def get_recent_rolling_averages():
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features", version=1, primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi", time_travel_format="HUDI",
    )
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp")

    latest_time = df["timestamp"].max()

    return {
        "aqi_avg_24h": df[df["timestamp"] >= latest_time - pd.Timedelta(hours=24)]["aqi"].mean(),
        "aqi_avg_48h": df[df["timestamp"] >= latest_time - pd.Timedelta(hours=48)]["aqi"].mean(),
        "aqi_avg_72h": df[df["timestamp"] >= latest_time - pd.Timedelta(hours=72)]["aqi"].mean(),
    }


def load_model(horizon_name):
    # log in to Hopsworks so we can reach the Model Registry
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    mr = project.get_model_registry()

    # ask the registry for this horizon's best model by name
    hw_model = mr.get_best_model(name=f"aqi_best_model_{horizon_name}", metric="rmse", direction="min")

    # download() pulls that model's files into a temp folder and returns the path to it
    download_dir = Path(hw_model.download())

    model = joblib.load(download_dir / "best_model.joblib")

    with open(download_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    scaler = None
    if metadata.get("needs_scaler"):
        scaler = joblib.load(download_dir / "scaler.joblib")

    return model, scaler, metadata



def get_live_row():
    #converts city name to long lat coordinates
    lat, lon = geocode(config.CITY_NAME, config.COUNTRY_CODE)
    #gets us the current (temp, humidity, wind)
    weather = get_current_weather(lat, lon)
    #fetch current pollution levels (pm2.5, pm10, co, no2, nh3)
    pollution = get_air_pollution(lat, lon)
    #combines both API responses into one clean row
    row = build_feature_row(config.CITY_NAME, weather, pollution)

    #save the raw row first - matches the feature store's actual schema
    save_row(row)

    #build a separate copy with rolling averages added, just for prediction use
    prediction_row = row.copy()
    rolling_averages = get_recent_rolling_averages()
    prediction_row.update(rolling_averages)

    return prediction_row


def predict_aqi(row, model, scaler, metadata):
     feature_columns= metadata["feature_columns"]   #it fetches columns of data from metadat(dictionary)
     x = pd.DataFrame([row])[feature_columns]  # converts the fetched data from rows to column(table form)

     if scaler is not None: # if scaler is still none as we assumed on line 19 no chnages needed and if it is not none means a converter is required 
               
                x= scaler.transform(x) #it converts the data x and overwrites it


     prediction = model.predict(x)[0]  #it predicts and loads the first from the list to the prediction
     return round(float(prediction), 1)   #converts the decimal value to one place



def check_hazard_alerts(row, predictions):
    alerts = []

    if row["aqi"] >= config.HAZARD_ALERT_THRESHOLD:
        alerts.append(f"Current AQI ({row['aqi']}) has reached hazardous levels")

    for horizon_name in ["24h", "48h", "72h"]:
        value = predictions[horizon_name]["value"]
        if value >= config.HAZARD_ALERT_THRESHOLD:
            alerts.append(f"{horizon_name} forecast ({value}) is expected to reach hazardous levels")

    return alerts


def log_hazard_alerts(alerts):
    if not alerts:
        return

    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="hazard_alerts",
        version=1,
        primary_key=["timestamp"],
        description="Log of triggered AQI hazard alerts",
        time_travel_format="HUDI",
    )

    row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "alert_count": len(alerts),
        "alert_message": " | ".join(alerts),
    }
    row_df = pd.DataFrame([row])
    fg.insert(row_df)
    print(f"Logged {len(alerts)} hazard alert(s) to Hopsworks.") 



if __name__ == "__main__":
    row = get_live_row()
    predictions = {}
    for horizon_name in ["24h", "48h", "72h"]:
        model, scaler, metadata = load_model(horizon_name)
        prediction = predict_aqi(row, model, scaler, metadata)
        predictions[horizon_name] = {"value": prediction, "model_used": metadata["best_model"]}
        print(f"{horizon_name} ahead prediction: {prediction} (model used: {metadata['best_model']})")
    print(f"Live AQI (from current PM2.5): {row['aqi']}")
    alerts = check_hazard_alerts(row, predictions)
    if alerts:
        print("\n⚠️  HAZARD ALERT")
        for alert in alerts:
            print(f"  - {alert}")
        log_hazard_alerts(alerts)