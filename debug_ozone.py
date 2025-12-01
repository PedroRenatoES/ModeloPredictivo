from src.data_processing import load_data, process_data
from src.config import get_features_for_target, PROCESSED_DATA_PATH
import pandas as pd

print("=" * 80)
print("DEBUGGING OZONE PROCESSING")
print("=" * 80)

# Load and process
df = load_data()
print(f"\n1. Loaded data: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

df_proc = process_data(df, target_name='ozone', is_training=True)
print(f"\n2. Processed data: {df_proc.shape}")
print(f"   Ozone columns: {[c for c in df_proc.columns if 'ozone' in c]}")

# Get required features
features = get_features_for_target('ozone')
print(f"\n3. Required features ({len(features)}):")
for f in features:
    exists = f in df_proc.columns
    print(f"   {'✓' if exists else '✗'} {f}")

# Check if we can train
missing = [f for f in features if f not in df_proc.columns]
if missing:
    print(f"\n❌ MISSING FEATURES: {missing}")
else:
    print(f"\n✓ All features present!")
    
    # Test loading processed data
    df_proc.to_csv(PROCESSED_DATA_PATH.replace('train_data.csv', 'train_data_ozone.csv'), index=False)
    df_loaded = pd.read_csv(PROCESSED_DATA_PATH.replace('train_data.csv', 'train_data_ozone.csv'))
    
    print(f"\n4. Saved and reloaded: {df_loaded.shape}")
    X_test = df_loaded[features]
    print(f"   Can create feature matrix: {X_test.shape}")
    print("\n✓ Ready for training!")
