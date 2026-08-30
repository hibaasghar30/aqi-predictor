# 🌫️ Karachi AQI Forecasting System

An end-to-end machine learning pipeline that predicts the Air Quality Index (AQI) for Karachi, Pakistan up to 72 hours in advance — built on a fully automated, serverless MLOps stack.

**🔗 Live dashboard:** [aqi-predictor-jhpqumq2pnd6fud4vuzzfj.streamlit.app](https://aqi-predictor-jhpqumq2pnd6fud4vuzzfj.streamlit.app/)

---

## What this project does

This system fetches live weather and pollution data every hour, engineers features from it, trains and evaluates multiple machine learning models daily, and serves 24h / 48h / 72h AQI forecasts through an interactive, publicly deployed web dashboard — complete with explainable predictions, health guidance, and hazard alerts.

It was built end-to-end: data collection → feature store → model training → model registry → automated CI/CD → live dashboard, with no manual step required to keep it running.

---

## Features

- **Live 24h / 48h / 72h AQI forecasts**, using the best-performing model per horizon (selected automatically from Ridge, Random Forest, and XGBoost based on RMSE)
- **SHAP-based explainability** — every forecast is paired with a chart showing which features are driving that specific prediction
- **Plain-language health guidance** translating the current AQI into concrete advice for the general public
- **"Best time for outdoor activity"** — automatically finds the lowest-AQI window across the current reading and all three forecasts
- **Hazard alerts** — a prominent warning banner when AQI crosses a dangerous threshold, logged for later review
- **Monthly AQI trend chart** with a year selector, and a live pollutant comparison chart
- **Fully automated pipeline** — new data collected every hour, models retrained daily, with zero manual intervention

---

## Architecture

```
 Weather & Pollution API (OpenWeatherMap)
              │
              ▼
   Feature Engineering  ──────►  Hopsworks Feature Store
              │                          │
              │                          ▼
              │                  Model Training (Ridge /
              │                  Random Forest / XGBoost)
              │                          │
              │                          ▼
              │                  Hopsworks Model Registry
              │                          │
              ▼                          ▼
        GitHub Actions  ───────►  Streamlit Dashboard
     (hourly + daily runs)          (Streamlit Cloud)
```

| Pipeline | What it does | Runs |
|---|---|---|
| Feature pipeline | Fetches live weather + pollutant data, engineers features, writes to the feature store | Every hour |
| Training pipeline | Trains & evaluates 3 models per horizon, registers the best one | Daily, 6:00 AM |
| Web application | Loads the latest model + features, generates and displays forecasts | On-demand |

---

## Tech stack

| Component | Tool |
|---|---|
| Weather & pollution data | [OpenWeatherMap API](https://openweathermap.org/api) |
| Historical weather backfill | [Open-Meteo Archive API](https://open-meteo.com/) |
| Feature store & model registry | [Hopsworks](https://www.hopsworks.ai/) (free tier) |
| Modelling | scikit-learn (Ridge, Random Forest), XGBoost |
| Explainability | [SHAP](https://shap.readthedocs.io/) |
| Automation / CI-CD | GitHub Actions |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Deployment | Streamlit Community Cloud |

---

## Project structure

```
aqi-predictor/
├── app/
│   └── streamlit_app.py       # Dashboard entry point
├── src/
│   ├── config.py              # AQI category thresholds, paths, API keys
│   ├── openweather_client.py  # Live + historical API calls
│   ├── aqi_util.py            # PM2.5 → AQI conversion
│   ├── feature_engineering.py # Builds a single feature row
│   ├── feature_store.py       # Hopsworks feature store connection
│   ├── predictor.py           # Live prediction + hazard alerts (GitHub Actions entry point)
│   ├── train_model.py         # Trains & registers models for all 3 horizons
│   ├── backfill.py            # One-time historical data backfill
│   ├── eda.py                 # Exploratory data analysis charts
│   └── drift_check.py         # Data drift monitoring
├── .github/workflows/
│   └── pipeline.yml           # Hourly + daily automation schedule
├── data/                      # EDA output charts
├── models/                    # Locally cached model files per horizon
└── requirements.txt
```

---

## Model performance

Three models — Ridge Regression, Random Forest, and XGBoost — are trained independently for each of the three forecast horizons. The model with the lowest RMSE on a held-out test set is automatically selected and registered for that horizon.

Forecast accuracy naturally decreases as the horizon lengthens (R² of ~0.86 at 24h vs. ~0.35 at 72h), which is expected — predicting further into the future is a harder problem with more accumulated uncertainty. This was reviewed with the project mentor and confirmed to be acceptable, provided the model beats a naive persistence baseline and RMSE stays reasonably low, both of which hold here.

Tree-based ensemble methods (Random Forest, XGBoost) were prioritized over deep learning, since they're widely regarded as a strong — often superior — choice for structured, tabular data of this size, whereas deep learning typically needs substantially larger datasets to outperform them.

---

## Running it locally

```bash
# Clone the repo
git clone https://github.com/hibaasghar30/aqi-predictor.git
cd aqi-predictor

# Install dependencies
pip install -r requirements.txt

# Set up your .env file with:
#   OPENWEATHER_API_KEY
#   HOPSWORKS_API_KEY
#   HOPSWORKS_PROJECT_NAME

# Run the dashboard
streamlit run app/streamlit_app.py
```

To manually trigger a live prediction + hazard alert check (the same script GitHub Actions runs hourly):

```bash
python -m src.predictor
```

---

## Automation

The entire pipeline runs on a schedule via GitHub Actions ([`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml)):

```yaml
on:
  schedule:
    - cron: "0 * * * *"    # every hour  -> feature pipeline
    - cron: "0 6 * * *"    # once daily  -> training + drift check
```

---

## Known limitations

- No deep learning model was trained — see the reasoning under **Model performance** above.
- RMSE, MAE, and R² are currently printed during training but not yet persisted to model metadata or surfaced on the dashboard.
- The free-tier Hopsworks account has a capped monthly compute budget; if exceeded, the automated hourly/daily pipeline pauses until the usage cycle resets, though the deployed dashboard continues serving existing data and predictions unaffected.

---

## Author

**Hiba Asghar**
[LinkedIn](https://www.linkedin.com/in/hiba-asghar-87a8683b7) · [GitHub](https://github.com/hibaasghar30)
