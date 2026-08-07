import requests
import pandas as pd
from datetime import datetime, timedelta
from src.openweather_client import get_air_pollution_history
from src.aqi_util import pm25_to_aqi
from src import config
from src.feature_store import get_feature_store

def fetch_historical_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "auto",
    }
    response = requests.get(url, params=params)
    data = response.json()

    weather_df = pd.DataFrame({
        "timestamp": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "pressure": data["hourly"]["surface_pressure"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })
    print(f"Fetched {len(weather_df)} hourly weather rows from Open-Meteo")
    return weather_df


#testing
#df = fetch_historical_weather(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
#print(df.head())
#print(df.tail())


def fetch_historical_pollution(lat, lon , start_date, end_date):
    start_dt =  datetime.strptime(start_date , "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_rows= []
    current = start_dt


    while current < end_dt:
        chunk_end = min(current + timedelta(days=30), end_dt)

        start_unix = int(current.timestamp())
        end_unix = int(chunk_end.timestamp())

        print(f"Fetching pollution: {current.date()} to {chunk_end.date()}")

        data = get_air_pollution_history(lat, lon, start_unix, end_unix)

        for entry in data.get("list", []):
            components = entry["components"]
            all_rows.append({
                "timestamp": datetime.fromtimestamp(entry["dt"]).isoformat(),
                "pm2_5": components["pm2_5"],
                "pm10": components["pm10"],
                "co": components["co"],
                "no2": components["no2"],
                "so2": components["so2"],
                "o3": components["o3"],
            })

        current = chunk_end


        pollution_df = pd.DataFrame(all_rows)
    print(f"Fetched {len(pollution_df)} hourly pollution rows from OpenWeather")
    return pollution_df


#testing
#df = fetch_historical_pollution(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
#print(df.head())
#print(df.tail())



#to get data at exact matching moments.For this pandas has a built in tool as merge_asof

def merge_weather_and_pollution(weather_df, pollution_df):
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    pollution_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
#these to lines convert timestaamp from plain text to realdate time

    weather_df  = weather_df.sort_values("timestamp")
    pollution_df= pollution_df.sort_values("timestamp")
    #sort timestamps in order from earliest to latest

    merged = pd.merge_asof(
        pollution_df , weather_df,
        on="timestamp", 
        direction="nearest",  #finds the closest weather in row not the exact match
        tolerance=pd.Timedelta("1h") #dontmatch if its more than 1hr
    )

    merged = merged.dropna() #if any row cant find a matching row within that 1h tolerance then itll drop those rows bec they must have some missing weather columns
    print(f"Merged into {len(merged)} rows with both weather and pollution data")
    return merged



#testing
#weather_df = fetch_historical_weather(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
#pollution_df = fetch_historical_pollution(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
#merged_df = merge_weather_and_pollution(weather_df, pollution_df)
#print(merged_df.head())
#print(f"Total merged rows: {len(merged_df)}")


#took 2 separate tables(weather, pollution)lined them up so each row now has both weather and pollution data for the same hour



def add_aqi_and_change_rate(df):
#ensures rows are in time order & renumber rows cleanly 
    df = df.sort_values("timestamp").reset_index(drop=True)

#runs func pm25toaqi on each row & saves result in new column "aqi"
    df["aqi"]= df["pm2_5"].apply(pm25_to_aqi)  

#calculate how many hours passed since the last row
    df["hours_elapsed"] = df["timestamp"].diff().dt.total_seconds() / 3600

#calc how much aqi changed since the last row
    df["aqi_change_rate"] = df["aqi"].diff() / df["hours_elapsed"]

#first row has no row before it (null) so this chnages the null value to 0.0
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0.0) 

#removes helper column (it was only needed to calc the rate)
    df= df.drop(columns=["hours_elapsed"])

    print(f"Calculated AQI and change rate for {len(df)} rows")  #confirm how many rows were processed
    return df  

#testing
weather_df = fetch_historical_weather(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
pollution_df = fetch_historical_pollution(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
merged_df = merge_weather_and_pollution(weather_df, pollution_df)
final_df = add_aqi_and_change_rate(merged_df)
print(final_df.head())
print(final_df[["timestamp", "aqi", "aqi_change_rate"]].head(10))

def upload_to_hopsworks(df):
    df["city"] = "Karachi"

    #add the time-component columns your live pipeline already includes
    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["day"] = df["timestamp"].dt.day.astype("int64")
    df["month"] = df["timestamp"].dt.month.astype("int64")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int64")

    #pressure needs to be a whole number, not a decimal
    df["pressure"] = df["pressure"].round().astype(int)

    df["timestamp"] = df["timestamp"].astype(str)

    fs = get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi",
        time_travel_format="HUDI",
    )

    fg.insert(df)
    print(f"Uploaded {len(df)} rows to Hopsworks feature store.")
#testing
#weather_df = fetch_historical_weather(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
#pollution_df = fetch_historical_pollution(24.8546842, 67.0207055, "2024-07-31", "2026-07-31")
#merged_df = merge_weather_and_pollution(weather_df, pollution_df)
#final_df = add_aqi_and_change_rate(merged_df)
#upload_to_hopsworks(final_df)