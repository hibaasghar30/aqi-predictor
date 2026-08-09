import json
from datetime import datetime, timedelta
import pandas as pd
import hopsworks
from src import config


def load_data():
    # same login/feature-store pattern used elsewhere in the project
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features", version=1, primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi", time_travel_format="HUDI",
    )
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    print(f"Loaded {len(df)} rows for drift check")
    return df


def compute_baseline_stats(df):
    # "what normal looks like" - mean and spread across the FULL historical dataset
    return {
        "pm2_5_mean": df["pm2_5"].mean(),
        "pm2_5_std": df["pm2_5"].std(),
        "aqi_mean": df["aqi"].mean(),
        "aqi_std": df["aqi"].std(),
    }


def compute_recent_stats(df, days=7):
    # "what's happening right now" - only the last N days of rows
    cutoff = df["timestamp"].max() - timedelta(days=days)
    recent_df = df[df["timestamp"] >= cutoff]
    print(f"Comparing against {len(recent_df)} rows from the last {days} days")
    return {
        "pm2_5_mean": recent_df["pm2_5"].mean(),
        "pm2_5_std": recent_df["pm2_5"].std(),
        "aqi_mean": recent_df["aqi"].mean(),
        "aqi_std": recent_df["aqi"].std(),
    }


def compare_and_report(baseline, recent, threshold_stds=2):
    report = {"timestamp": datetime.now().isoformat(), "checks": {}}
    drift_detected = False
    row = {"timestamp": datetime.now().isoformat()}

    for column in ["pm2_5", "aqi"]:
        baseline_mean = baseline[f"{column}_mean"]
        baseline_std = baseline[f"{column}_std"]
        recent_mean = recent[f"{column}_mean"]

        distance = abs(recent_mean - baseline_mean) / baseline_std
        is_drifted = distance > threshold_stds

        report["checks"][column] = {
            "baseline_mean": round(baseline_mean, 2),
            "recent_mean": round(recent_mean, 2),
            "distance_in_stds": round(distance, 2),
            "drift_detected": bool(is_drifted),
        }

        # flatten these same numbers into the row we'll save to Hopsworks
        row[f"{column}_baseline_mean"] = round(baseline_mean, 2)
        row[f"{column}_recent_mean"] = round(recent_mean, 2)
        row[f"{column}_distance_in_stds"] = round(distance, 2)
        row[f"{column}_drift_detected"] = bool(is_drifted)

        if is_drifted:
            drift_detected = True
            print(f"⚠️ DRIFT DETECTED in {column}: recent avg {recent_mean:.2f} vs "
                  f"baseline avg {baseline_mean:.2f} ({distance:.2f} std devs away)")
        else:
            print(f"OK: {column} recent avg {recent_mean:.2f} vs baseline avg "
                  f"{baseline_mean:.2f} ({distance:.2f} std devs away)")

    report["drift_detected"] = drift_detected
    row["drift_detected"] = drift_detected

    # save today's report as one row in a permanent Hopsworks feature group (history over time)
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    drift_fg = fs.get_or_create_feature_group(
        name="drift_reports", version=1, primary_key=["timestamp"],
        description="Daily data drift check results", online_enabled=False,
        time_travel_format="HUDI",
    )
    row_df = pd.DataFrame([row])
    row_df["timestamp"] = row_df["timestamp"].astype(str)
    drift_fg.insert(row_df)

    print("Drift report saved to Hopsworks feature group 'drift_reports'.")
    return report


#testing
df = load_data()
baseline = compute_baseline_stats(df)
recent = compute_recent_stats(df)
compare_and_report(baseline, recent)