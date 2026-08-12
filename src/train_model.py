#what this file does
#it opens our saved data and two different methods try to learn the pattern
#  and saves the pattern that learnedthe pattern well is used by the predictor
import hopsworks
import json   # saves notes about the winning model
import pandas as pd   #opens and reads our saved feature store data as a table
from sklearn.model_selection import train_test_split       # splits data into "practice" and "quiz" portions, so testing is fair
from sklearn.linear_model import Ridge                       # METHOD 1: one way of learning the aqi pattern
from sklearn.ensemble import RandomForestRegressor            # METHOD 2: a different way of learning the aqi pattern
from sklearn.preprocessing import StandardScaler               # adjusts numbers to a similar scale - Ridge needs this to learn well
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from xgboost import XGBRegressor
  # STEP 3: three ways to score how good each method's guesses were
#model predicted AQI as 80, but the real AQI was 75. That's a mistake of 5.
#   1.  mean_absolute_error (MAE)
#Takes every mistake, ignores whether it was too high or too low, and just averages how far off 
#you were on average. If your model is consistently off by  5 points across all its predictions, MAE ≈ 5.



#     2.    mean_squared_error (used to get RMSE)
#Similar idea, but squares each mistake before averaging. Squaring a mistake of 5 gives 25;
#  squaring a mistake of 20 gives 400. This means big mistakes get punished much more harshly than small ones


#     3.     r2 score
#  instead of measuring the size of mistakes it measures what percentage of the pattern your  
#  model actually captured. It's a number from roughly 0 to 1:
#1= best 0= not good just guessing avg , negative= worst



import joblib                        # STEP 4: saves the winning method's learned pattern to a file
from src import config               # tells this file where to find our data, and where to save results




#opens file"feature_store" and loads the rows we have collected so far
def load_training_data():
    #log into hopsworks and get the feature store
    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
#get the same feature group we've been saving to
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["city", "timestamp"],
        description="AQI and weather features for Karachi, with rolling averages and multi-horizon targets",
        time_travel_format="HUDI",
    )

    df = fg.read()
    print(f"Loaded {len(df)} rows from the feature store (version 2)")
    return df




#builds three file paths (model file, metadata file, scaler file) with the horizon name baked into each filename, e.g. models/best_model_24h.joblib — so the 24h, 48h, and 72h models each get their own distinct files instead of overwriting one shared file.
def get_model_paths(horizon_name):
    horizon_dir = config.MODEL_DIR / horizon_name
    horizon_dir.mkdir(exist_ok=True)

    model_file = horizon_dir / "best_model.joblib"
    metadata_file = horizon_dir / "model_metadata.json"
    scaler_file = horizon_dir / "scaler.joblib"
    return model_file, metadata_file, scaler_file



#--------------------------     TESTING    -------------------------
#testing
#df = load_training_data()
#print(df)


def train_one_horizon(df, horizon_name, target_column):
    print(f"\n===== Training {horizon_name} model (target: {target_column}) =====")

    feature_columns = [
        "pm2_5", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity",
        "pressure", "wind_speed", "hour", "day", "month", "day_of_week",
        "aqi", "aqi_avg_24h", "aqi_avg_48h", "aqi_avg_72h",
    ]

    # only drop rows missing something THIS horizon actually needs
    needed_columns = feature_columns + [target_column]
    horizon_df = df.dropna(subset=needed_columns)
    print(f"{len(horizon_df)} / {len(df)} rows usable for {horizon_name} (after dropping incomplete ones)")

    x = horizon_df[feature_columns]
    y = horizon_df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #----------------------------------      TRAINING RIDGE      -----------------------------
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ridge = Ridge()
    ridge.fit(x_train_scaled, y_train)
    ridge_predictions = ridge.predict(x_test_scaled)
    ridge_rmse = mean_squared_error(y_test, ridge_predictions) ** 0.5
    ridge_mae = mean_absolute_error(y_test, ridge_predictions)
    ridge_r2 = r2_score(y_test, ridge_predictions)
    print(f"Ridge - RMSE: {ridge_rmse:.2f}, MAE: {ridge_mae:.2f}, R2: {ridge_r2:.2f}")

    #----------------------------------      TRAINING RANDOM FOREST      -----------------------------
    forest = RandomForestRegressor(random_state=42)
    forest.fit(x_train, y_train)
    forest_predictions = forest.predict(x_test)
    forest_rmse = mean_squared_error(y_test, forest_predictions) ** 0.5
    forest_mae = mean_absolute_error(y_test, forest_predictions)
    forest_r2 = r2_score(y_test, forest_predictions)
    print(f"Random Forest - RMSE: {forest_rmse:.2f}, MAE: {forest_mae:.2f}, R2: {forest_r2:.2f}")

    #----------------------------------      TRAINING XGBOOST      -----------------------------
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(x_train, y_train)
    xgb_predictions = xgb_model.predict(x_test)
    xgb_rmse = mean_squared_error(y_test, xgb_predictions) ** 0.5
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_r2 = r2_score(y_test, xgb_predictions)
    print(f"XGBoost - RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}, R2: {xgb_r2:.2f}")

    #compare all three models - lower RMSE wins
    candidates = {
        "ridge": (ridge, ridge_rmse),
        "random_forest": (forest, forest_rmse),
        "xgboost": (xgb_model, xgb_rmse),
    }
    best_name = min(candidates, key=lambda name: candidates[name][1])
    best_model = candidates[best_name][0]
    best_rmse = candidates[best_name][1]
    needs_scaler = (best_name == "ridge")

    print(f"Winner for {horizon_name}: {best_name}")

    model_file, metadata_file, scaler_file = get_model_paths(horizon_name)

    joblib.dump(best_model, model_file)
    if needs_scaler:
        joblib.dump(scaler, scaler_file)

    metadata = {
        "horizon": horizon_name,
        "best_model": best_name,
        "needs_scaler": needs_scaler,
        "feature_columns": feature_columns,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    print(f"{horizon_name} model and metadata saved.")

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(
        name=f"aqi_best_model_{horizon_name}",
        metrics={"rmse": best_rmse},
        description=f"Best AQI model for {horizon_name} horizon - {best_name}",
    )
    hw_model.save(str(model_file.parent))
    #wrong method: hw_model.save(str(config.MODEL_DIR))
    print(f"{horizon_name} model uploaded to Hopsworks Model Registry as 'aqi_best_model_{horizon_name}'.")

def train_and_evaluate():
    df = load_training_data()
    train_one_horizon(df, "24h", "aqi_24h_ahead")
    train_one_horizon(df, "48h", "aqi_48h_ahead")
    train_one_horizon(df, "72h", "aqi_72h_ahead")


train_and_evaluate()