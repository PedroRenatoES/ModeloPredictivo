import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys
from src.config import PROCESSED_DATA_PATH, MODELS_DIR, FEATURES, TEST_SIZE, HORIZONS

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "Corr": corr}

def train_model():
    print("Loading processed data...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"{PROCESSED_DATA_PATH} not found. Run data_processing.py first.")
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Time-series split
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    results = []
    
    print("\n" + "="*80)
    print("  STARTING MULTI-HORIZON TRAINING")
    print("="*80)

    for h in HORIZONS:
        target_col = f"target_{h}h"
        print(f"\n>>> Training for Horizon: {h}h ({target_col})")
        
        X_train = train_df[FEATURES]
        y_train = train_df[target_col]
        X_test = test_df[FEATURES]
        y_test = test_df[target_col]
        
        # Initialize XGBoost
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
        
        # Train with reduced verbosity to avoid freezing IDE
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False 
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Baseline (Persistence: predict t+h using t)
        # For h=1, we use t (pm2_5). For h=12, we use t (pm2_5).
        # Persistence assumption: Future value will be same as current value.
        y_baseline = test_df["pm2_5"] 
        
        metrics_model = calculate_metrics(y_test, y_pred)
        metrics_base = calculate_metrics(y_test, y_baseline)
        
        # Skill Score
        skill = (1 - metrics_model["MAE"] / metrics_base["MAE"]) * 100
        
        results.append({
            "Horizon": f"pm2_5_{h}h",
            "MAE": metrics_model["MAE"],
            "RMSE": metrics_model["RMSE"],
            "R2": metrics_model["R2"],
            "MAPE": metrics_model["MAPE"],
            "Corr": metrics_model["Corr"],
            "Skill": skill,
            "Base_MAE": metrics_base["MAE"]
        })
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"xgboost_pm25_{h}h.json")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

    # Print Final Report
    print("\n" + "="*100)
    print("  MÉTRICAS EN TEST SET (DATOS NO VISTOS)")
    print("="*100)
    print(f"{'Horizonte':<12} | {'MAE':<8} | {'RMSE':<8} | {'R2':<8} | {'MAPE':<8} | {'Corr':<8} | {'Skill':<8}")
    print("-" * 90)
    
    for res in results:
        print(f"{res['Horizon']:<12} | {res['MAE']:<8.3f} | {res['RMSE']:<8.3f} | {res['R2']:<8.3f} | {res['MAPE']:<8.2f}% | {res['Corr']:<8.3f} | {res['Skill']:>+8.2f}%")
    
    print("="*100 + "\n")
    
    return results

if __name__ == "__main__":
    train_model()
