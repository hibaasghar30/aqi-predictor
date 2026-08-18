import hopsworks
from src import config

project = hopsworks.login(
    api_key_value=config.HOPSWORKS_API_KEY,
    project=config.HOPSWORKS_PROJECT_NAME,
)
mr = project.get_model_registry()

for horizon_name in ["24h", "48h", "72h"]:
    model_name = f"aqi_best_model_{horizon_name}"
    models = mr.get_models(name=model_name)
    models.sort(key=lambda m: m.version, reverse=True)

    print(f"\n{model_name}: found {len(models)} versions")
    keep = models[:3]
    delete = models[3:]

    print(f"Keeping versions: {[m.version for m in keep]}")
    print(f"Deleting versions: {[m.version for m in delete]}")

    for m in delete:
        m.delete()
        print(f"Deleted version {m.version}")
        