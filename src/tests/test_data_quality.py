"""
Test suite for data quality validation.

Tests cover:
- Temporal ordering
- Duplicate detection
- Null value handling
- Value range validation
- Physical coherence
"""
import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_data, process_data
from src.config import RAW_DATA_PATH


class TestTemporalOrdering:
    """Test temporal ordering of data."""
    
    def test_time_column_sorted(self, raw_data):
        """Test that time column is sorted ascending."""
        time_col = raw_data["time"]
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(time_col):
            time_col = pd.to_datetime(time_col)
        
        # Check if sorted
        is_sorted = time_col.is_monotonic_increasing
        assert is_sorted, "Time column is not sorted in ascending order"
    
    def test_processed_data_maintains_order(self, raw_data):
        """Test that processed data maintains temporal order."""
        processed = process_data(raw_data.copy(), target_name="pm2_5", is_training=False)
        
        time_col = pd.to_datetime(processed["time"])
        assert time_col.is_monotonic_increasing, \
            "Processed data lost temporal ordering"
    
    def test_no_backward_time_jumps(self, raw_data):
        """Test that there are no backward time jumps."""
        time_col = pd.to_datetime(raw_data["time"])
        time_diff = time_col.diff()
        
        # Skip first row (will be NaT)
        negative_jumps = (time_diff[1:] < pd.Timedelta(0)).sum()
        
        assert negative_jumps == 0, \
            f"Found {negative_jumps} backward time jumps in data"


class TestDuplicates:
    """Test for duplicate timestamps."""
    
    def test_no_duplicate_timestamps(self, raw_data):
        """Test that there are no duplicate timestamps in raw data."""
        time_col = raw_data["time"]
        duplicates = time_col.duplicated().sum()
        
        assert duplicates == 0, \
            f"Found {duplicates} duplicate timestamps in raw data"
    
    def test_processed_data_no_duplicates(self, raw_data):
        """Test that processed data has no duplicate timestamps."""
        processed = process_data(raw_data.copy(), target_name="pm2_5", is_training=False)
        
        duplicates = processed["time"].duplicated().sum()
        
        assert duplicates == 0, \
            f"Found {duplicates} duplicate timestamps in processed data"


class TestNullValues:
    """Test null value handling."""
    
    def test_processed_features_no_nulls(self, raw_data):
        """Test that processed features have no null values."""
        from src.config import get_features_for_target
        
        target = "pm2_5"
        processed = process_data(raw_data.copy(), target_name=target, is_training=False)
        features = get_features_for_target(target)
        
        # Check for nulls in features
        for feature in features:
            if feature in processed.columns:
                null_count = processed[feature].isna().sum()
                assert null_count == 0, \
                    f"Feature '{feature}' has {null_count} null values after processing"
    
    def test_training_data_no_nulls(self, raw_data):
        """Test that training data has no nulls after dropna."""
        # Use smaller sample for faster testing
        sample = raw_data.head(1000).copy()
        processed = process_data(sample, target_name="pm2_5", is_training=True)
        
        # Training data should have no nulls anywhere
        null_counts = processed.isna().sum()
        total_nulls = null_counts.sum()
        
        assert total_nulls == 0, \
            f"Training data has {total_nulls} null values after processing"


class TestValueRanges:
    """Test that values are within physically reasonable ranges."""
    
    def test_pm25_positive(self, raw_data):
        """Test that PM2.5 values are positive."""
        pm25_col = raw_data["pm2_5"]
        negative_count = (pm25_col < 0).sum()
        
        assert negative_count == 0, \
            f"Found {negative_count} negative PM2.5 values"
    
    def test_pm10_positive(self, raw_data):
        """Test that PM10 values are positive."""
        pm10_col = raw_data["pm10"]
        negative_count = (pm10_col < 0).sum()
        
        assert negative_count == 0, \
            f"Found {negative_count} negative PM10 values"
    
    def test_ozone_positive(self, raw_data):
        """Test that ozone values are positive."""
        ozone_col = raw_data["ozone"]
        negative_count = (ozone_col < 0).sum()
        
        assert negative_count == 0, \
            f"Found {negative_count} negative ozone values"
    
    def test_nitrogen_dioxide_positive(self, raw_data):
        """Test that NO₂ values are positive."""
        no2_col = raw_data["nitrogen_dioxide"]
        negative_count = (no2_col < 0).sum()
        
        assert negative_count == 0, \
            f"Found {negative_count} negative NO₂ values"
    
    def test_temperature_reasonable_range(self, raw_data):
        """Test that temperature is in reasonable range for Santa Cruz, Bolivia."""
        temp_col = raw_data["temperature_2m"]
        
        # Santa Cruz temperatures typically range from 5°C to 40°C
        # Allow some margin for extreme events
        too_cold = (temp_col < -10).sum()
        too_hot = (temp_col > 45).sum()
        
        assert too_cold == 0, \
            f"Found {too_cold} unreasonably cold temperatures (<-10°C)"
        assert too_hot == 0, \
            f"Found {too_hot} unreasonably hot temperatures (>45°C)"
    
    def test_humidity_valid_range(self, raw_data):
        """Test that relative humidity is between 0 and 100%."""
        humidity_col = raw_data["relative_humidity_2m"]
        
        below_zero = (humidity_col < 0).sum()
        above_hundred = (humidity_col > 100).sum()
        
        assert below_zero == 0, \
            f"Found {below_zero} humidity values below 0%"
        assert above_hundred == 0, \
            f"Found {above_hundred} humidity values above 100%"
    
    def test_wind_speed_non_negative(self, raw_data):
        """Test that wind speed is non-negative."""
        wind_col = raw_data["wind_speed_10m"]
        
        negative_count = (wind_col < 0).sum()
        
        assert negative_count == 0, \
            f"Found {negative_count} negative wind speed values"
    
    def test_wind_direction_valid_range(self, raw_data):
        """Test that wind direction is between 0 and 360 degrees."""
        wind_dir_col = raw_data["wind_direction_10m"]
        
        below_zero = (wind_dir_col < 0).sum()
        above_360 = (wind_dir_col > 360).sum()
        
        assert below_zero == 0, \
            f"Found {below_zero} wind direction values below 0°"
        assert above_360 == 0, \
            f"Found {above_360} wind direction values above 360°"
    
    def test_precipitation_non_negative(self, raw_data):
        """Test that precipitation is non-negative."""
        precip_col = raw_data["precipitation"]
        
        negative_count = (precip_col < 0).sum()
        
        assert negative_count == 0, \
            f"Found {negative_count} negative precipitation values"
    
    def test_pressure_reasonable_range(self, raw_data):
        """Test that surface pressure is in reasonable range."""
        pressure_col = raw_data["surface_pressure"]
        
        # Typical sea-level pressure is ~1013 hPa
        # Santa Cruz is ~400m elevation, so expect ~960-1030 hPa
        # Allow wider range for extreme weather
        too_low = (pressure_col < 900).sum()
        too_high = (pressure_col > 1100).sum()
        
        assert too_low == 0, \
            f"Found {too_low} unreasonably low pressure values (<900 hPa)"
        assert too_high == 0, \
            f"Found {too_high} unreasonably high pressure values (>1100 hPa)"


class TestPhysicalCoherence:
    """Test physical coherence between related variables."""
    
    def test_pm10_greater_than_pm25(self, raw_data):
        """Test that PM10 >= PM2.5 (PM2.5 is a subset of PM10)."""
        # Filter out rows with nulls
        valid_rows = raw_data[["pm2_5", "pm10"]].dropna()
        
        if len(valid_rows) > 0:
            violations = (valid_rows["pm10"] < valid_rows["pm2_5"]).sum()
            violation_pct = violations / len(valid_rows) * 100
            
            # Allow small percentage of violations due to measurement errors
            assert violation_pct < 5, \
                f"PM10 < PM2.5 in {violation_pct:.1f}% of cases (should be <5%)"
    
    def test_wind_components_magnitude(self, raw_data):
        """Test that wind components reconstruct original magnitude."""
        processed = process_data(raw_data.copy(), target_name="pm2_5", is_training=False)
        
        # Reconstruct wind speed from U/V components
        reconstructed = np.sqrt(processed["wind_u"]**2 + processed["wind_v"]**2)
        original = raw_data["wind_speed_10m"].values[:len(reconstructed)]
        
        # Should match within numerical precision
        max_error = np.abs(reconstructed - original).max()
        
        assert max_error < 0.5, \
            f"Wind vectorization inaccurate: max error = {max_error:.3f} m/s"
    
    def test_cyclical_features_normalized(self, raw_data):
        """Test that cyclical features maintain sin²+cos²=1."""
        processed = process_data(raw_data.copy(), target_name="pm2_5", is_training=False)
        
        # Hour cyclical
        hour_norm = processed["hour_sin"]**2 + processed["hour_cos"]**2
        assert np.allclose(hour_norm, 1.0, atol=0.01), \
            "Hour cyclical features not normalized"
        
        # Month cyclical
        month_norm = processed["month_sin"]**2 + processed["month_cos"]**2
        assert np.allclose(month_norm, 1.0, atol=0.01), \
            "Month cyclical features not normalized"


class TestDataCompleteness:
    """Test data completeness and coverage."""
    
    def test_sufficient_data_volume(self, raw_data):
        """Test that we have sufficient data for training."""
        # Should have at least 1000 rows for meaningful training
        assert len(raw_data) >= 1000, \
            f"Insufficient data: {len(raw_data)} rows (need at least 1000)"
    
    def test_all_required_columns_present(self, raw_data):
        """Test that all required columns are present."""
        required_columns = [
            "time", "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "precipitation", "surface_pressure"
        ]
        
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        
        assert len(missing_columns) == 0, \
            f"Missing required columns: {missing_columns}"
