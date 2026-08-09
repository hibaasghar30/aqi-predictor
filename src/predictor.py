import joblib           #saves trained model file
import json
import pandas as pd
from src import config
from src.openweather_client import geocode, get_current_weather, get_air_pollution
from src.feature_engineering import build_feature_row
import hopsworks
from pathlib import Path

def load_model():
    # log in to Hopsworks so we can reach the Model Registry
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    mr = project.get_model_registry()

    # ask the registry for whichever uploaded "aqi_best_model" has the lowest rmse
    hw_model = mr.get_best_model(name="aqi_best_model", metric="rmse", direction="min")

    # download() pulls that model's files into a temp folder and returns the path to it
    download_dir = Path(hw_model.download())

    # load everything from that downloaded folder instead of local models/
    model = joblib.load(download_dir / "best_model.joblib")

    with open(download_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    scaler = None
    if metadata.get("needs_scaler"):
        scaler = joblib.load(download_dir / "scaler.joblib")

    return model, scaler, metadata
def get_live_row():
#conerts cityname to long lat coordiantes
        lat, lon = geocode(config.CITY_NAME, config.COUNTRY_CODE)
#gets us the current  (temp, humidity, wind)
        weather = get_current_weather(lat, lon)
#fetch current pollution levels (pm2.5, pm10, co, no2 , nh3)
        pollution= get_air_pollution(lat, lon)
#combines both API responses into one clean row
        row= build_feature_row(config.CITY_NAME, weather, pollution)
        return row





#testing 
#row = get_live_row()
#print(row)





def predict_aqi(row, model, scaler, metadata):
     feature_columns= metadata["feature_columns"]   #it fetches columns of data from metadat(dictionary)
     x = pd.DataFrame([row])[feature_columns]  # converts the fetched data from rows to column(table form)

     if scaler is not None: # if scaler is still none as we assumed on line 19 no chnages needed and if it is not none means a converter is required 
               
                x= scaler.transform(x) #it converts the data x and overwrites it


     prediction = model.predict(x)[0]  #it predicts and loads the first from the list to the prediction
     return round(prediction, 1)   #converts the decimal value to one place

#testing full pipeline
model, scaler, metadata = load_model()
row = get_live_row()
prediction = predict_aqi(row, model, scaler, metadata)

print(f"Live AQI (from current PM2.5): {row['aqi']}")
print(f"Model prediction: {prediction}")
print(f"Model used: {metadata['best_model']}")