import sys
import os
from src.data_processing import load_data, process_data, save_data
from src.train import train_model

def main():
    print("Starting AirQuality_Pro Pipeline...")
    
    # 1. Data Processing
    try:
        df = load_data()
        df = process_data(df)
        save_data(df)
    except Exception as e:
        print(f"Error in data processing: {e}")
        sys.exit(1)
        
    # 2. Training
    try:
        results = train_model()
    except Exception as e:
        print(f"Error in training: {e}")
        sys.exit(1)
        
    print("\nPipeline completed successfully!")
    
    # Check 1h horizon R2
    r2_1h = next((r["R2"] for r in results if r["Horizon"] == "pm2_5_1h"), 0)
    
    if r2_1h < 0.5:
        print(f"WARNING: 1-hour R2 score is low ({r2_1h:.4f}). Consider tuning hyperparameters.")
    else:
        print(f"Model performance for 1h looks promising (R2: {r2_1h:.4f}).")

if __name__ == "__main__":
    main()
