import pandas as pd
from src.feature_store import get_feature_store


def load_all_data():
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
         name="aqi_features", version = 1, primary_key=["city" , "timestamp"],
         description = "AQI and weather features for Karachi", time_travel_format = "HUDI",
    )

    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format ="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows to build horizon features")
    return df

def add_rolling_features(df):
    # rolling() needs a time-based index to understand "past 24 hours" as real time, not row count
    df = df.set_index("timestamp")
    df["aqi_avg_24h"] = df["aqi"].rolling("24h").mean()
    df["aqi_avg_48h"] = df["aqi"].rolling("48h").mean()
    df["aqi_avg_72h"] = df["aqi"].rolling("72h").mean()
    df = df.reset_index()
    print("Added rolling average features (24h, 48h, 72h)")
    return df

def add_horizon_target(df, hours_ahead, tolerance_minutes=30):
    # build a lookup table of just timestamp + aqi, shifted backward in time
    lookup = df[["timestamp", "aqi"]].copy()
    lookup["timestamp"] = lookup["timestamp"] - pd.Timedelta(hours=hours_ahead)

    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        lookup.sort_values("timestamp"),
        on="timestamp",
        direction="forward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        suffixes=("", f"_{hours_ahead}h_ahead"),
    )
    return merged[f"aqi_{hours_ahead}h_ahead"]


def build_and_save():
    df = load_all_data()
    df = add_rolling_features(df)

    df["aqi_24h_ahead"] = add_horizon_target(df, hours_ahead=24)
    df["aqi_48h_ahead"] = add_horizon_target(df, hours_ahead=48)
    df["aqi_72h_ahead"] = add_horizon_target(df, hours_ahead=72)

    print(f"Rows with a valid 24h target: {df['aqi_24h_ahead'].notna().sum()} / {len(df)}")
    print(f"Rows with a valid 48h target: {df['aqi_48h_ahead'].notna().sum()} / {len(df)}")
    print(f"Rows with a valid 72h target: {df['aqi_72h_ahead'].notna().sum()} / {len(df)}")

    df["timestamp"] = df["timestamp"].astype(str)

    fs = get_feature_store()
    fg_v2 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi, with rolling averages and multi-horizon targets",
        time_travel_format="HUDI",
    )
    fg_v2.insert(df)
    print("Saved enriched data to aqi_features version 2.")


build_and_save()