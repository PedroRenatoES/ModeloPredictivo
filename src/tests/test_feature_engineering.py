"""
Test suite for feature engineering functionality.

Tests cover:
- Lag features (t-1, t-24)
- Rolling statistics (mean, std over 24h)
- Temporal cyclical features (Fourier transforms for hour/month)
- Wind vectorization (U/V components)
"""
import pytest
import pandas as pd
import numpy as np
from src.data_processing import process_data, load_data
from src.config import AVAILABLE_TARGETS


class TestLagFeatures:
    """Test lag feature creation."""
    
    def test_lag_1_calculation(self, raw_data):
        """Test that lag_1 is correctly calculated (t-1)."""
        target = "pm2_5"
        df = raw_data.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        # Process data to create lag features
        processed = process_data(df, target_name=target, is_training=False)
        
        # Check that lag_1 exists
        lag_col = f"{target}_lag_1"
        assert lag_col in processed.columns, f"Missing {lag_col}"
        
        # Verify lag_1 is shifted by 1 position
        # lag_1 at position i should equal target at position i-1
        # Skip first row (will be NaN after processing)
        if len(processed) > 1:
            original_val = df[target].iloc[5]  # Pick a middle value
            lag_val = processed[lag_col].iloc[6]  # Next row's lag
            assert abs(original_val - lag_val) < 0.01, "lag_1 not correctly shifted"
    
    def test_lag_24_calculation(self, raw_data):
        """Test that lag_24 is correctly calculated (t-24)."""
        target = "pm2_5"
        df = raw_data.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        processed = process_data(df, target_name=target, is_training=False)
        
        # Check that lag_24 exists
        lag_col = f"{target}_lag_24"
        assert lag_col in processed.columns, f"Missing {lag_col}"
        
        # Verify lag_24 is shifted by 24 positions
        if len(processed) > 30:
            original_val = df[target].iloc[30]
            lag_val = processed[lag_col].iloc[54]  # 30 + 24
            assert abs(original_val - lag_val) < 0.01, "lag_24 not correctly shifted"
    
    def test_all_targets_have_lags(self):
        """Test that all targets get lag features."""
        # Create simple test data
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
        
        for target in AVAILABLE_TARGETS:
            processed = process_data(test_data.copy(), target_name=target, is_training=False)
            assert f"{target}_lag_1" in processed.columns, f"Missing lag_1 for {target}"
            assert f"{target}_lag_24" in processed.columns, f"Missing lag_24 for {target}"


class TestRollingStatistics:
    """Test rolling statistics features."""
    
    def test_rolling_mean_24(self, raw_data):
        """Test 24-hour rolling mean calculation."""
        target = "pm2_5"
        df = raw_data.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        processed = process_data(df, target_name=target, is_training=False)
        
        rolling_col = f"{target}_rolling_mean_24"
        assert rolling_col in processed.columns, f"Missing {rolling_col}"
        
        # Check that rolling mean is calculated (should not be all NaN)
        assert processed[rolling_col].notna().any(), "Rolling mean is all NaN"
        
        # Verify rolling mean is reasonable (between min and max of target)
        valid_means = processed[rolling_col].dropna()
        if len(valid_means) > 0:
            target_min = df[target].min()
            target_max = df[target].max()
            assert valid_means.min() >= target_min - 1, "Rolling mean below target min"
            assert valid_means.max() <= target_max + 1, "Rolling mean above target max"
    
    def test_rolling_std_24(self, raw_data):
        """Test 24-hour rolling standard deviation calculation."""
        target = "pm2_5"
        df = raw_data.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        processed = process_data(df, target_name=target, is_training=False)
        
        rolling_col = f"{target}_rolling_std_24"
        assert rolling_col in processed.columns, f"Missing {rolling_col}"
        
        # Rolling std should be >= 0
        valid_stds = processed[rolling_col].dropna()
        if len(valid_stds) > 0:
            assert (valid_stds >= 0).all(), "Rolling std has negative values"
    
    def test_all_targets_have_rolling_stats(self):
        """Test that all targets get rolling statistics."""
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
        
        for target in AVAILABLE_TARGETS:
            processed = process_data(test_data.copy(), target_name=target, is_training=False)
            assert f"{target}_rolling_mean_24" in processed.columns
            assert f"{target}_rolling_std_24" in processed.columns


class TestCyclicalFeatures:
    """Test temporal cyclical (Fourier) features."""
    
    def test_hour_cyclical_features(self, raw_data):
        """Test hour sin/cos features."""
        df = raw_data.copy()
        processed = process_data(df, target_name="pm2_5", is_training=False)
        
        # Check presence
        assert "hour_sin" in processed.columns, "Missing hour_sin"
        assert "hour_cos" in processed.columns, "Missing hour_cos"
        
        # Check range [-1, 1]
        assert processed["hour_sin"].between(-1, 1).all(), "hour_sin out of range"
        assert processed["hour_cos"].between(-1, 1).all(), "hour_cos out of range"
        
        # Check periodicity: sin^2 + cos^2 = 1
        cyclical_sum = processed["hour_sin"]**2 + processed["hour_cos"]**2
        assert np.allclose(cyclical_sum, 1.0, atol=0.01), "Hour cyclical not normalized"
    
    def test_month_cyclical_features(self, raw_data):
        """Test month sin/cos features."""
        df = raw_data.copy()
        processed = process_data(df, target_name="pm2_5", is_training=False)
        
        # Check presence
        assert "month_sin" in processed.columns, "Missing month_sin"
        assert "month_cos" in processed.columns, "Missing month_cos"
        
        # Check range [-1, 1]
        assert processed["month_sin"].between(-1, 1).all(), "month_sin out of range"
        assert processed["month_cos"].between(-1, 1).all(), "month_cos out of range"
        
        # Check periodicity
        cyclical_sum = processed["month_sin"]**2 + processed["month_cos"]**2
        assert np.allclose(cyclical_sum, 1.0, atol=0.01), "Month cyclical not normalized"


class TestWindVectorization:
    """Test wind component vectorization."""
    
    def test_wind_components_exist(self, raw_data):
        """Test that wind_u and wind_v are created."""
        df = raw_data.copy()
        processed = process_data(df, target_name="pm2_5", is_training=False)
        
        assert "wind_u" in processed.columns, "Missing wind_u component"
        assert "wind_v" in processed.columns, "Missing wind_v component"
    
    def test_wind_magnitude_preserved(self, raw_data):
        """Test that wind magnitude is preserved after vectorization."""
        df = raw_data.copy()
        processed = process_data(df, target_name="pm2_5", is_training=False)
        
        # Reconstruct magnitude from components
        reconstructed_speed = np.sqrt(processed["wind_u"]**2 + processed["wind_v"]**2)
        
        # Should match original wind_speed_10m
        original_speed = df["wind_speed_10m"].values[:len(reconstructed_speed)]
        
        # Allow small numerical errors
        assert np.allclose(reconstructed_speed, original_speed, atol=0.1), \
            "Wind magnitude not preserved in vectorization"
    
    def test_wind_vector_known_directions(self):
        """Test wind vectorization with known directions."""
        # Create test data with known wind directions
        test_data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=4, freq="h"),
            "pm2_5": [20, 20, 20, 20],
            "pm10": [30, 30, 30, 30],
            "ozone": [40, 40, 40, 40],
            "nitrogen_dioxide": [15, 15, 15, 15],
            "temperature_2m": [25, 25, 25, 25],
            "relative_humidity_2m": [60, 60, 60, 60],
            "wind_speed_10m": [10, 10, 10, 10],  # Constant speed
            "wind_direction_10m": [0, 90, 180, 270],  # N, E, S, W
            "precipitation": [0, 0, 0, 0],
            "surface_pressure": [1013, 1013, 1013, 1013]
        })
        
        processed = process_data(test_data, target_name="pm2_5", is_training=False)
        
        # North (0°): U=0, V=10
        assert abs(processed["wind_u"].iloc[0]) < 0.1, "North wind U component wrong"
        assert abs(processed["wind_v"].iloc[0] - 10) < 0.1, "North wind V component wrong"
        
        # East (90°): U=10, V=0
        assert abs(processed["wind_u"].iloc[1] - 10) < 0.1, "East wind U component wrong"
        assert abs(processed["wind_v"].iloc[1]) < 0.1, "East wind V component wrong"
