import pandas as pd
import hopsworks
from src import config


def get_feature_store():
    #logs into your hopsworks project using the API key from .env
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    #gets access to that project's feature store
    fs = project.get_feature_store()
    return fs

#this function adds new rows to all the previous saved rows
def save_row(row):
    fs = get_feature_store()

    #get (or create, if it doesn't exist yet) a feature group called "aqi_features"
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi",
        time_travel_format="HUDI",
    )

    #wrap the row in a DataFrame, same as before
    newrow_df = pd.DataFrame([row])

    #insert this row into the feature group
    fg.insert(newrow_df)
    print("Saved row to Hopsworks feature store.")


def get_last_row():
    fs = get_feature_store()

    #get the same feature group we save to
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi",
        time_travel_format="HUDI",
    )

    #read all data currently in the feature group
    df = fg.read()

    if len(df) == 0:
        return None

    #sort by timestamp so the most recent row is last, then grab it
    df = df.sort_values("timestamp")
    return df.iloc[-1].to_dict()



#testing get_last_row with hopsworks
last = get_last_row()
print(last)













#additional info    parquet is same as csv but more efficient for large datasets
#using functions to ******TEST************ IN FEATURE_ENGINEERINGFILE ON LINE 70


#testing connection
#fs = get_feature_store()
#print("Connected to Hopsworks feature store successfully!")
#testing save_row with hopsworks
#test_row = {
    #"city": "Karachi",
    #"pm2_5": 25.0,
    #"pm10": 100.0,
    #"co": 60.0,
    #"no2": 0.1,
    #"so2": 0.5,
    #"o3": 40.0,
   # "temperature": 30.0,
    #"humidity": 70,
    #"pressure": 1000,
   # "wind_speed": 5.0,
  #  "hour": 12,
   # "day": 25,
    #"month": 7,
    #"day_of_week": 5,
   # "aqi": 80.0,
  #  "aqi_change_rate": 0.0,
 #   "timestamp": "2026-07-25T12:00:00",
#}
#save_row(test_row)