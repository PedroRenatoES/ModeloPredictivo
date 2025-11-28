import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODELS_DIR, FEATURES, RAW_DATA_PATH, HORIZONS
from src.data_processing import process_data

def load_models():
    models = {}
    for h in HORIZONS:
        model_path = os.path.join(MODELS_DIR, f"xgboost_pm25_{h}h.json")
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            models[h] = model
    return models

def predict_manual(current_data):
    """
    current_data: dict containing:
        - time (str or datetime)
        - pm2_5
        - nitrogen_dioxide
        - ozone
        - temperature_2m
        - relative_humidity_2m
        - wind_speed_10m
        - wind_direction_10m
        - precipitation
        - surface_pressure
    """
    print("Loading historical data for context...")
    # Load raw data to get history
    df_history = pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])
    
    # Create DataFrame from input
    input_df = pd.DataFrame([current_data])
    input_df["time"] = pd.to_datetime(input_df["time"])
    
    # Append input to history
    # We need history to calculate lags and rolling means
    full_df = pd.concat([df_history, input_df], ignore_index=True)
    full_df = full_df.sort_values("time").reset_index(drop=True)
    
    # Process data (calculates all features: wind_u, lags, rolling, etc.)
    # We suppress print output from process_data to keep it clean
    print("Processing features...")
    processed_df = process_data(full_df, is_training=False)
    
    # Get the processed row for our input time
    # It should be the last row (or close to it if dates are sorted)
    target_time = input_df["time"].iloc[0]
    target_row = processed_df[processed_df["time"] == target_time]
    
    if target_row.empty:
        print("Error: Could not process features for the given time. Maybe it's too old?")
        return

    X_input = target_row[FEATURES]
    
    # Load models
    print("Loading models...")
    models = load_models()
    
    if not models:
        print("No models found. Please train first.")
        return {}

    predictions = {}
    for h in HORIZONS:
        if h in models:
            pred = models[h].predict(X_input)[0]
            # Convert numpy float to python float for JSON serialization
            predictions[f"{h}h"] = float(pred)
            
    return predictions

def print_predictions(predictions, target_time, current_pm25):
    print(f"\n{'='*60}")
    print(f"PREDICTION REPORT")
    print(f"Input Time: {target_time}")
    print(f"Current PM2.5: {current_pm25}")
    print(f"{'='*60}")
    print(f"{'Horizon':<15} | {'Predicted Time':<25} | {'Prediction':<15}")
    print(f"{'-'*60}")
    
    for h_str, pred in predictions.items():
        h = int(h_str.replace("h", ""))
        pred_time = target_time + pd.Timedelta(hours=h)
        print(f"+{h:<2} hours       | {str(pred_time):<25} | {pred:.2f}")
            
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Example Usage:
    current_conditions = {
        "time": datetime.now(),
        "pm2_5": 15.5,
        "nitrogen_dioxide": 20.0,
        "ozone": 45.0,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 5.5,
        "wind_direction_10m": 180.0,
        "precipitation": 0.0,
        "surface_pressure": 1013.0
    }
    
    preds = predict_manual(current_conditions)
    if preds:
        print_predictions(preds, current_conditions["time"], current_conditions["pm2_5"])
