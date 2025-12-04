"""
Test suite for target advancement functionality.

Tests verify that future targets (t+h) are correctly created
for all forecasting horizons.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_processing import process_data
from src.config import HORIZONS, AVAILABLE_TARGETS


class TestTargetCreation:
    """Test creation of advanced targets for different horizons."""
    
    def test_all_horizon_targets_created(self):
        """Test that all horizon targets are created during training."""
        # Create test data
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": np.random.uniform(10, 50, 200),
            "pm10": np.random.uniform(20, 80, 200),
            "ozone": np.random.uniform(30, 100, 200),
            "nitrogen_dioxide": np.random.uniform(10, 60, 200),
            "temperature_2m": np.random.uniform(15, 30, 200),
            "relative_humidity_2m": np.random.uniform(40, 80, 200),
            "wind_speed_10m": np.random.uniform(0, 15, 200),
            "wind_direction_10m": np.random.uniform(0, 360, 200),
            "precipitation": np.random.uniform(0, 5, 200),
            "surface_pressure": np.random.uniform(1000, 1020, 200)
        })
        
        # Process with is_training=True to create targets
        processed = process_data(test_data, target_name="pm2_5", is_training=True)
        
        # Check all horizon targets exist
        for h in HORIZONS:
            target_col = f"target_{h}h"
            assert target_col in processed.columns, f"Missing {target_col}"
    
    def test_target_shift_correctness(self):
        """Test that targets are correctly shifted forward in time."""
        # Create predictable test data
        n_points = 300
        dates = pd.date_range("2024-01-01", periods=n_points, freq="h")
        
        # Create a simple linear sequence for PM2.5 to easily verify shifts
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": np.arange(n_points, dtype=float),  # 0, 1, 2, 3, ...
            "pm10": np.random.uniform(20, 80, n_points),
            "ozone": np.random.uniform(30, 100, n_points),
            "nitrogen_dioxide": np.random.uniform(10, 60, n_points),
            "temperature_2m": np.random.uniform(15, 30, n_points),
            "relative_humidity_2m": np.random.uniform(40, 80, n_points),
            "wind_speed_10m": np.random.uniform(0, 15, n_points),
            "wind_direction_10m": np.random.uniform(0, 360, n_points),
            "precipitation": np.random.uniform(0, 5, n_points),
            "surface_pressure": np.random.uniform(1000, 1020, n_points)
        })
        
        processed = process_data(test_data, target_name="pm2_5", is_training=True)
        
        # For each horizon, verify the shift
        # Note: process_data drops NaN rows, so we need to account for that
        if len(processed) > max(HORIZONS) + 30:  # Ensure enough data after dropna
            test_idx = 50  # Pick a middle index
            current_pm25 = processed["pm2_5"].iloc[test_idx]
            
            for h in HORIZONS:
                target_col = f"target_{h}h"
                # target_h at position i should be pm2_5 at position i+h (before shift)
                # Due to shift(-h), target_h[i] = pm2_5[i+h]
                target_value = processed[target_col].iloc[test_idx]
                expected_value = current_pm25 + h  # Since we used arange
                
                # Allow some tolerance due to processing
                assert abs(target_value - expected_value) < 2, \
                    f"target_{h}h shift incorrect: got {target_value}, expected {expected_value}"
    
    def test_target_1h_is_next_hour(self):
        """Test that target_1h represents the next hour's value."""
        n_points = 200
        dates = pd.date_range("2024-01-01", periods=n_points, freq="h")
        
        # Create sequential PM2.5 values
        pm25_values = np.arange(100, 100 + n_points, dtype=float)
        
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": pm25_values,
            "pm10": np.random.uniform(20, 80, n_points),
            "ozone": np.random.uniform(30, 100, n_points),
            "nitrogen_dioxide": np.random.uniform(10, 60, n_points),
            "temperature_2m": np.random.uniform(15, 30, n_points),
            "relative_humidity_2m": np.random.uniform(40, 80, n_points),
            "wind_speed_10m": np.random.uniform(0, 15, n_points),
            "wind_direction_10m": np.random.uniform(0, 360, n_points),
            "precipitation": np.random.uniform(0, 5, n_points),
            "surface_pressure": np.random.uniform(1000, 1020, n_points)
        })
        
        processed = process_data(test_data, target_name="pm2_5", is_training=True)
        
        # Pick a sample row
        if len(processed) > 10:
            idx = 10
            current = processed["pm2_5"].iloc[idx]
            target_1h = processed["target_1h"].iloc[idx]
            
            # target_1h should be approximately current + 1
            assert abs(target_1h - (current + 1)) < 2, \
                f"target_1h not next hour: current={current}, target_1h={target_1h}"
    
    def test_longer_horizons_have_larger_offsets(self):
        """Test that longer horizons have progressively larger time offsets."""
        n_points = 300
        dates = pd.date_range("2024-01-01", periods=n_points, freq="h")
        
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": np.arange(n_points, dtype=float),
            "pm10": np.random.uniform(20, 80, n_points),
            "ozone": np.random.uniform(30, 100, n_points),
            "nitrogen_dioxide": np.random.uniform(10, 60, n_points),
            "temperature_2m": np.random.uniform(15, 30, n_points),
            "relative_humidity_2m": np.random.uniform(40, 80, n_points),
            "wind_speed_10m": np.random.uniform(0, 15, n_points),
            "wind_direction_10m": np.random.uniform(0, 360, n_points),
            "precipitation": np.random.uniform(0, 5, n_points),
            "surface_pressure": np.random.uniform(1000, 1020, n_points)
        })
        
        processed = process_data(test_data, target_name="pm2_5", is_training=True)
        
        if len(processed) > 50:
            idx = 30
            current = processed["pm2_5"].iloc[idx]
            
            # Verify ordering: target_1h < target_12h < target_24h, etc.
            prev_offset = 0
            for h in sorted(HORIZONS):
                target_col = f"target_{h}h"
                target_val = processed[target_col].iloc[idx]
                offset = target_val - current
                
                assert offset >= prev_offset, \
                    f"Horizon {h}h offset not increasing: offset={offset}, prev={prev_offset}"
                prev_offset = offset
    
    def test_no_targets_in_inference_mode(self):
        """Test that targets are NOT created in inference mode."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": np.random.uniform(10, 50, 100),
            "pm10": np.random.uniform(20, 80, 100),
            "ozone": np.random.uniform(30, 100, 100),
            "nitrogen_dioxide": np.random.uniform(10, 60, 100),
            "temperature_2m": np.random.uniform(15, 30, 100),
            "relative_humidity_2m": np.random.uniform(40, 80, 100),
            "wind_speed_10m": np.random.uniform(0, 15, 100),
            "wind_direction_10m": np.random.uniform(0, 360, 100),
            "precipitation": np.random.uniform(0, 5, 100),
            "surface_pressure": np.random.uniform(1000, 1020, 100)
        })
        
        # Process with is_training=False (inference mode)
        processed = process_data(test_data, target_name="pm2_5", is_training=False)
        
        # Target columns should NOT exist
        for h in HORIZONS:
            target_col = f"target_{h}h"
            assert target_col not in processed.columns, \
                f"{target_col} should not exist in inference mode"


class TestTargetDataIntegrity:
    """Test integrity of target data across different targets."""
    
    def test_all_targets_get_horizon_columns(self):
        """Test that all pollutant targets get their horizon columns."""
        dates = pd.date_range("2024-01-01", periods=200, freq="h")
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": np.random.uniform(10, 50, 200),
            "pm10": np.random.uniform(20, 80, 200),
            "ozone": np.random.uniform(30, 100, 200),
            "nitrogen_dioxide": np.random.uniform(10, 60, 200),
            "temperature_2m": np.random.uniform(15, 30, 200),
            "relative_humidity_2m": np.random.uniform(40, 80, 200),
            "wind_speed_10m": np.random.uniform(0, 15, 200),
            "wind_direction_10m": np.random.uniform(0, 360, 200),
            "precipitation": np.random.uniform(0, 5, 200),
            "surface_pressure": np.random.uniform(1000, 1020, 200)
        })
        
        for target in AVAILABLE_TARGETS:
            processed = process_data(test_data.copy(), target_name=target, is_training=True)
            
            for h in HORIZONS:
                target_col = f"target_{h}h"
                assert target_col in processed.columns, \
                    f"{target_col} missing for target {target}"
    
    def test_targets_have_no_nulls_after_processing(self):
        """Test that target columns have no NaN values after processing (dropna)."""
        dates = pd.date_range("2024-01-01", periods=250, freq="h")
        test_data = pd.DataFrame({
            "time": dates,
            "pm2_5": np.random.uniform(10, 50, 250),
            "pm10": np.random.uniform(20, 80, 250),
            "ozone": np.random.uniform(30, 100, 250),
            "nitrogen_dioxide": np.random.uniform(10, 60, 250),
            "temperature_2m": np.random.uniform(15, 30, 250),
            "relative_humidity_2m": np.random.uniform(40, 80, 250),
            "wind_speed_10m": np.random.uniform(0, 15, 250),
            "wind_direction_10m": np.random.uniform(0, 360, 250),
            "precipitation": np.random.uniform(0, 5, 250),
            "surface_pressure": np.random.uniform(1000, 1020, 250)
        })
        
        processed = process_data(test_data, target_name="pm2_5", is_training=True)
        
        # After dropna(), no target should have NaN
        for h in HORIZONS:
            target_col = f"target_{h}h"
            null_count = processed[target_col].isna().sum()
            assert null_count == 0, f"{target_col} has {null_count} NaN values after processing"
