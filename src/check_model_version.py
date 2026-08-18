import hopsworks
from src import config

project = hopsworks.login(
    api_key_value=config.HOPSWORKS_API_KEY,
    project=config.HOPSWORKS_PROJECT_NAME,
)
mr = project.get_model_registry()
hw_model = mr.get_best_model(name="aqi_best_model_72h", metric="rmse", direction="min")
print(f"Currently selected version: {hw_model.version}")
print(f"RMSE: {hw_model.training_metrics}")
