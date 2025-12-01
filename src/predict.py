import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
import argparse
from src.config import MODELS_DIR, PROCESSED_DATA_PATH, HORIZONS, AVAILABLE_TARGETS, get_features_for_target

def load_models(target_name="pm2_5", horizons=None):
    """
    Load trained models for a specific target variable.
    
    Args:
        target_name (str): Name of the target variable
        horizons (list): List of horizons to load (default: all HORIZONS)
    
    Returns:
        dict: Dictionary mapping horizon to loaded model
    """
    if horizons is None:
        horizons = HORIZONS
    
    models = {}
    for h in horizons:
        model_path = os.path.join(MODELS_DIR, f"xgboost_{target_name}_{h}h.json")
        if not os.path.exists(model_path):
            print(f"⚠️  Warning: Model for {target_name} @ {h}h not found at {model_path}")
            continue
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models[h] = model
    return models

def predict_sample(target_name="pm2_5", data_suffix="", horizons=None):
    """
    Make predictions for a target variable using the most recent data point.
    
    Args:
        target_name (str): Name of the target variable to predict
        data_suffix (str): Suffix for processed data file (e.g., "_ozone")
        horizons (list): List of horizons to predict (default: all HORIZONS)
    """
    if horizons is None:
        horizons = HORIZONS
    
    print(f"\n{'='*80}")
    print(f"  PREDICTION FOR {target_name.upper()}")
    print(f"{'='*80}\n")
    
    print("Loading models...")
    models = load_models(target_name, horizons)
    
    if not models:
        print(f"❌ No models found for {target_name}. Run main.py to train first:")
        print(f"   python main.py --target {target_name}")
        return

    print("Loading data for inference...")
    
    # Try to load target-specific processed data first
    data_path = PROCESSED_DATA_PATH
    if data_suffix:
        base, ext = os.path.splitext(PROCESSED_DATA_PATH)
        data_path = f"{base}{data_suffix}{ext}"
    
    if not os.path.exists(data_path):
        print(f"⚠️  Target-specific data not found at {data_path}")
        print(f"   Trying default processed data path...")
        data_path = PROCESSED_DATA_PATH
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Get features for this target
    features = get_features_for_target(target_name)
    
    # Take the last row (most recent time step)
    # We want to predict t+1, t+12, etc. from this single point in time
    last_row = df.tail(1).copy()
    
    # Check if all required features exist
    missing_features = [f for f in features if f not in last_row.columns]
    if missing_features:
        print(f"❌ Missing features in data: {missing_features}")
        print(f"   Please ensure data is processed with target={target_name}")
        return
    
    X_sample = last_row[features]
    current_time = last_row["time"].values[0]
    current_value = last_row[target_name].values[0] if target_name in last_row.columns else "N/A"
    
    print(f"\n📍 Making predictions from time: {current_time}")
    print(f"📊 Current {target_name} value: {current_value}")
    print(f"🔮 Forecasting for horizons: {sorted(models.keys())} hours\n")
    
    print("="*90)
    print(f"{'Horizon':<15} | {'Predicted Time':<25} | {target_name.upper() + ' Prediction':<20} | {'Change':<15}")
    print("-" * 90)
    
    for h in sorted(models.keys()):
        pred = models[h].predict(X_sample)[0]
        
        # Calculate predicted time
        pred_time = pd.to_datetime(current_time) + pd.Timedelta(hours=h)
        
        # Calculate change from current value
        if current_value != "N/A":
            change = pred - current_value
            change_pct = (change / current_value * 100) if current_value != 0 else 0
            change_str = f"{change:+.2f} ({change_pct:+.1f}%)"
        else:
            change_str = "N/A"
        
        print(f"{h} hours ({h}h)    | {str(pred_time):<25} | {pred:<20.2f} | {change_str:<15}")
        
    print("="*90)
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Make predictions for air quality variables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict PM2.5 (default)
  python -m src.predict
  
  # Predict ozone
  python -m src.predict --target ozone
  
  # Predict NO₂ with specific horizons
  python -m src.predict --target nitrogen_dioxide --horizons 1,12,24
        """
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='pm2_5',
        choices=AVAILABLE_TARGETS,
        help=f'Target variable to predict (default: pm2_5)'
    )
    
    parser.add_argument(
        '--horizons',
        type=str,
        default=None,
        help=f'Comma-separated forecast horizons in hours (default: all available)'
    )
    
    args = parser.parse_args()
    
    # Parse horizons if provided
    horizons = None
    if args.horizons:
        try:
            horizons = [int(h.strip()) for h in args.horizons.split(',')]
        except ValueError:
            print("ERROR: Horizons must be comma-separated integers (e.g., 1,12,24)")
            sys.exit(1)
    
    predict_sample(
        target_name=args.target,
        data_suffix=f"_{args.target}",
        horizons=horizons
    )

if __name__ == "__main__":
    main()

