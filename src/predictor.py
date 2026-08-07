import joblib           #saves trained model file
import json
import pandas as pd
from src import config
from src.openweather_client import geocode, get_current_weather, get_air_pollution
from src.feature_engineering import build_feature_row


def load_model():
    model = joblib.load(config.MODEL_FILE)

# open lets python open the file same as double clicking
#r is a mode known as read , it allows to read the file
#f is just the short form of the file you opened
#json.load read the file and converts it into python dict and hands that dict to metadata
    with open(config.MODEL_METADATA_FILE ,"r") as f:
        metadata= json.load(f)

        scaler = None  #assumes that no converter is needed
        if metadata.get("needs_scaler"):   #checks if a scaler is actually needed
            scaler = joblib.load(config.MODEL_DIR/ "scaler.joblib")  #loads the data from this file
    return model , scaler , metadata
  # hands back the file that was opened , converter (if needed) and metadata (the stickynotes)

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