import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_data(path=RAW_DATA_PATH):
    """Loads raw data."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def process_data(df):
    """Applies cleaning and feature engineering."""
    print("Processing data...")
    
    # 1. Handle Missing Values
    # Forward fill for time series is often appropriate for short gaps
    df = df.ffill().bfill()
    
    # 2. Wind Vectorization
    # Convert speed and direction to U and V components
    # wind_direction_10m is likely in degrees
    wd_rad = df["wind_direction_10m"] * np.pi / 180
    df["wind_u"] = df["wind_speed_10m"] * np.cos(wd_rad)
    df["wind_v"] = df["wind_speed_10m"] * np.sin(wd_rad)
    
    # 3. Time Cyclical Features
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # 4. Lag Features (Past values)
    # We want to predict t using t-1, t-24, etc.
    df["pm2_5_lag_1"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24"] = df["pm2_5"].shift(24)
    
    # 5. Rolling Statistics
    # Rolling mean/std of the last 24 hours (excluding current)
    # shift(1) ensures we don't use current value in the rolling window calculation for the current step
    df["pm2_5_rolling_mean_24"] = df["pm2_5"].shift(1).rolling(window=24).mean()
    df["pm2_5_rolling_std_24"] = df["pm2_5"].shift(1).rolling(window=24).std()
    
    # 6. Multi-Horizon Targets
    # We want to predict pm2_5 at t+h. 
    # So we create columns 'target_{h}h' which are the pm2_5 values shifted BACKWARDS by h.
    from src.config import HORIZONS
    for h in HORIZONS:
        df[f"target_{h}h"] = df["pm2_5"].shift(-h)

    # Drop rows with NaNs (both from lags at the start and targets at the end)
    df = df.dropna().reset_index(drop=True)
    
    print(f"Data processed. Shape: {df.shape}")
    return df

def save_data(df, path=PROCESSED_DATA_PATH):
    """Saves processed data."""
    print(f"Saving processed data to {path}...")
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data()
    df = process_data(df)
    save_data(df)
