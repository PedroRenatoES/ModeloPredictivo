import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
from src.config import MODELS_DIR, FEATURES, PROCESSED_DATA_PATH, HORIZONS

def load_models():
    models = {}
    for h in HORIZONS:
        model_path = os.path.join(MODELS_DIR, f"xgboost_pm25_{h}h.json")
        if not os.path.exists(model_path):
            print(f"Warning: Model for {h}h not found at {model_path}")
            continue
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models[h] = model
    return models

def predict_sample():
    print("Loading models...")
    models = load_models()
    
    if not models:
        print("No models found. Run main.py to train first.")
        return

    print("Loading data for inference...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found.")
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Take the last row (most recent time step)
    # We want to predict t+1, t+12, etc. from this single point in time
    last_row = df.tail(1).copy()
    X_sample = last_row[FEATURES]
    current_time = last_row["time"].values[0]
    
    print(f"\nMaking predictions from time: {current_time}")
    print("="*60)
    print(f"{'Horizon':<15} | {'Predicted Time':<25} | {'PM2.5 Prediction':<15}")
    print("-" * 60)
    
    for h in HORIZONS:
        if h in models:
            pred = models[h].predict(X_sample)[0]
            
            # Calculate predicted time
            pred_time = pd.to_datetime(current_time) + pd.Timedelta(hours=h)
            
            print(f"{h} hours ({h}h)    | {str(pred_time):<25} | {pred:.2f}")
            
    print("="*60)

if __name__ == "__main__":
    predict_sample()
