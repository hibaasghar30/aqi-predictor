import pandas as pd
import matplotlib.pyplot as plt
import hopsworks
from src import config


def load_data():
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi",
        time_travel_format="HUDI",
    )

    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp")

    print(f"Loaded {len(df)} rows for analysis")
    return df

#testing
#df = load_data()
#print(df[["timestamp", "aqi"]])

def plot_aqi_trend(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["aqi"], marker="o")
    plt.title("AQI Over Time - Karachi")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(config.PROJECT_ROOT / "data" / "aqi_trend.png")
    print("Saved AQI trend chart to data/aqi_trend.png")

#testing
#df = load_data()
#plot_aqi_trend(df)


def plot_aqi_by_hour(df):
    plt.figure(figsize=(10, 5))
    plt.scatter(df["hour"], df["aqi"])
    plt.title("AQI by Hour of Day - Karachi")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("AQI")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(config.PROJECT_ROOT / "data" / "aqi_by_hour.png")
    print("Saved AQI-by-hour chart to data/aqi_by_hour.png")



def plot_correlations(df):
    feature_columns = ["pm2_5", "pm10", "co", "no2", "so2", "o3",
                        "temperature", "humidity", "pressure", "wind_speed"]

    correlations = df[feature_columns + ["aqi"]].corr()["aqi"].drop("aqi")

    plt.figure(figsize=(10, 5))
    correlations.plot(kind="bar")
    plt.title("Correlation Between Features and AQI")
    plt.xlabel("Feature")
    plt.ylabel("Correlation with AQI")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(config.PROJECT_ROOT / "data" / "aqi_correlations.png")
    print("Saved correlation chart to data/aqi_correlations.png")

    
#testing
df = load_data()
plot_aqi_trend(df)
plot_aqi_by_hour(df)
plot_correlations(df)