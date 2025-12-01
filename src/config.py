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
TARGET = "pm2_5"  # Default target, can be overridden

# Available target variables for prediction
AVAILABLE_TARGETS = [
    "pm2_5",
    "pm10", 
    "ozone",
    "nitrogen_dioxide"
]

# Legacy: kept for backwards compatibility
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

def get_features_for_target(target_name):
    """
    Generates the feature list for a given target variable.
    
    This function creates a dynamic feature list that:
    1. Includes all meteorological variables
    2. Includes all pollutants EXCEPT the target (prevents data leakage)
    3. Includes time-cyclical features
    4. Includes lag and rolling statistics for the specific target
    
    Args:
        target_name (str): Name of the target variable to predict
        
    Returns:
        list: List of feature column names
        
    Example:
        >>> get_features_for_target("ozone")
        # Returns features including pm2_5, nitrogen_dioxide, etc. but NOT ozone
        # Also includes ozone_lag_1, ozone_lag_24, ozone_rolling_mean_24, etc.
    """
    if target_name not in AVAILABLE_TARGETS:
        raise ValueError(f"Target '{target_name}' not in available targets: {AVAILABLE_TARGETS}")
    
    # Base meteorological features (always included)
    base_features = [
        "temperature_2m",
        "relative_humidity_2m", 
        "precipitation",
        "surface_pressure",
        "wind_u",
        "wind_v"
    ]
    
    # Time-cyclical features (always included)
    time_features = [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos"
    ]
    
    # All pollutants available in dataset
    all_pollutants = [
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "ozone"
    ]
    
    # Use other pollutants as features (but NOT the target itself)
    # This allows cross-pollutant relationships without data leakage
    pollutant_features = [p for p in all_pollutants if p != target_name]
    
    # Lag features specific to the target
    lag_features = [
        f"{target_name}_lag_1",
        f"{target_name}_lag_24"
    ]
    
    # Rolling statistics specific to the target
    rolling_features = [
        f"{target_name}_rolling_mean_24",
        f"{target_name}_rolling_std_24"
    ]
    
    # Combine all features
    all_features = (
        base_features + 
        time_features + 
        pollutant_features + 
        lag_features + 
        rolling_features
    )
    
    return all_features
