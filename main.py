import sys
import os
import argparse
from src.data_processing import load_data, process_data, save_data
from src.train import train_model
from src.config import AVAILABLE_TARGETS, HORIZONS

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train air quality prediction models for multiple pollutants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PM2.5 models (default)
  python main.py
  
  # Train ozone models with specific horizons
  python main.py --target ozone --horizons 1,12,24
  
  # Train NO₂ models with all horizons
  python main.py --target nitrogen_dioxide
  
  # Train CO models with long-term forecasts
  python main.py --target carbon_monoxide --horizons 24,72,168
        """
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='pm2_5',
        choices=AVAILABLE_TARGETS,
        help=f'Target variable to predict (default: pm2_5). Available: {", ".join(AVAILABLE_TARGETS)}'
    )
    
    parser.add_argument(
        '--horizons',
        type=str,
        default=','.join(map(str, HORIZONS)),
        help=f'Comma-separated forecast horizons in hours (default: {",".join(map(str, HORIZONS))})'
    )
    
    args = parser.parse_args()
    
    # Parse horizons
    try:
        horizons = [int(h.strip()) for h in args.horizons.split(',')]
    except ValueError:
        print("ERROR: Horizons must be comma-separated integers (e.g., 1,12,24)")
        sys.exit(1)
    
    target = args.target
    
    print("="*80)
    print("  AIR QUALITY PREDICTION - MULTI-TARGET SYSTEM")
    print("="*80)
    print(f"  Target Variable: {target.upper()}")
    print(f"  Forecast Horizons: {horizons} hours")
    print("="*80 + "\n")
    
    # 1. Data Processing
    try:
        print("STEP 1: DATA PROCESSING")
        print("-" * 80)
        df = load_data()
        df = process_data(df, target_name=target, is_training=True)
        save_data(df, suffix=f"_{target}")
        print("✓ Data processing completed\n")
    except Exception as e:
        print(f"✗ Error in data processing: {e}")
        sys.exit(1)
        
    # 2. Training
    try:
        print("STEP 2: MODEL TRAINING")
        print("-" * 80)
        results = train_model(target_name=target, horizons=horizons)
        print("✓ Model training completed\n")
    except Exception as e:
        print(f"✗ Error in training: {e}")
        sys.exit(1)
        
    print("\n" + "="*80)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Performance summary
    if results:
        best_r2 = max(r["R2"] for r in results)
        worst_r2 = min(r["R2"] for r in results)
        avg_r2 = sum(r["R2"] for r in results) / len(results)
        
        print(f"\n📊 Performance Summary for {target.upper()}:")
        print(f"  - Best R²: {best_r2:.4f}")
        print(f"  - Worst R²: {worst_r2:.4f}")
        print(f"  - Average R²: {avg_r2:.4f}")
        
        # Check 1h horizon if it exists
        r2_1h = next((r["R2"] for r in results if "1h" in r["Horizon"]), None)
        
        if r2_1h is not None:
            if r2_1h < 0.5:
                print(f"\n⚠️  WARNING: 1-hour R² score is low ({r2_1h:.4f}). Consider:")
                print("  - Collecting more training data")
                print("  - Tuning hyperparameters")
                print("  - Adding more relevant features")
            else:
                print(f"\n✓ Model performance for 1h looks promising (R²: {r2_1h:.4f})")
    
    print(f"\n💾 Models saved in: models/xgboost_{target}_*h.json")
    print(f"📁 Processed data saved in: data/processed/train_data_{target}.csv\n")

if __name__ == "__main__":
    main()

