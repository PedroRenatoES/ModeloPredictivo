import os

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
#RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset_pm_scz_2013_2025.csv")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset_pm_scz_2022_2025.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "train_data.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_pm25.json")

# Model Parameters
TARGET = "pm2_5"
FEATURES = [
    "nitrogen_dioxide", "ozone", "temperature_2m", "relative_humidity_2m",
    "precipitation", "surface_pressure",
    "wind_u", "wind_v",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "pm2_5_lag_1", "pm2_5_lag_24",
    "pm2_5_rolling_mean_24", "pm2_5_rolling_std_24"
]

# Training Parameters
HORIZONS = [1, 12, 24, 72, 168]
TEST_SIZE = 0.2
RANDOM_STATE = 42
