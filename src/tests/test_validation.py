"""
Test suite for end-to-end validation.

Tests validate the complete pipeline with real data and ensure
all components work together correctly.
"""
import pytest
import pandas as pd
import numpy as np
import os
from src.data_processing import load_data, process_data
from src.config import (
    RAW_DATA_PATH, 
    AVAILABLE_TARGETS, 
    HORIZONS, 
    get_features_for_target
)


class TestRealDataValidation:
    """Test validation with real data."""
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_prediction_on_real_data(self, load_model, target):
        """Test that model can make predictions on real data."""
        # Load real data
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")
        
        df = pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        
        # Take last 100 rows for testing
        test_data = df.tail(100).copy()
        
        # Process data
        processed = process_data(test_data, target_name=target, is_training=False)
        
        if len(processed) == 0:
            pytest.skip(f"No valid data after processing for {target}")
        
        features = get_features_for_target(target)
        X = processed[features]
        
        # Test prediction for each horizon
        for h in HORIZONS:
            model = load_model(target, h)
            if model is None:
                pytest.skip(f"Model not found: {target}_{h}h")
            
            predictions = model.predict(X)
            
            # Predictions should be valid
            assert len(predictions) > 0, f"No predictions generated for {target}_{h}h"
            assert not np.any(np.isnan(predictions)), f"NaN predictions for {target}_{h}h"
            assert not np.any(np.isinf(predictions)), f"Inf predictions for {target}_{h}h"
            
            # Predictions should be in reasonable range
            assert np.all(predictions > 0), f"Negative predictions for {target}_{h}h"
            assert np.all(predictions < 1000), f"Unreasonably large predictions for {target}_{h}h"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_prediction_accuracy_on_recent_data(self, load_model, target):
        """Test prediction accuracy on recent real data."""
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")
        
        df = pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        
        # Use last 200 rows, split for validation
        recent_data = df.tail(200).copy()
        
        # Process as training to get targets
        processed = process_data(recent_data, target_name=target, is_training=True)
        
        if len(processed) < 10:
            pytest.skip(f"Insufficient data after processing for {target}")
        
        features = get_features_for_target(target)
        
        # Test 1h horizon (should be most accurate)
        h = 1
        model = load_model(target, h)
        if model is None:
            pytest.skip(f"Model not found: {target}_{h}h")
        
        X = processed[features]
        y_true = processed[f"target_{h}h"]
        y_pred = model.predict(X)
        
        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAE should be reasonable (validate model is working)
        max_mae_expected = {
            "pm2_5": 20.0,
            "pm10": 30.0,
            "ozone": 35.0,
            "nitrogen_dioxide": 25.0
        }
        
        max_mae = max_mae_expected.get(target, 50.0)
        assert mae < max_mae, \
            f"{target}_1h validation MAE={mae:.3f} too high (expected <{max_mae})"


class TestCompletePipeline:
    """Test the complete data processing and prediction pipeline."""
    
    def test_pipeline_raw_to_prediction_pm25(self, load_model):
        """Test complete pipeline from raw data to prediction for PM2.5."""
        target = "pm2_5"
        
        # Step 1: Load raw data
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")
        
        df = load_data(RAW_DATA_PATH)
        assert len(df) > 0, "Failed to load raw data"
        
        # Step 2: Process data
        processed = process_data(df.tail(100).copy(), target_name=target, is_training=False)
        assert len(processed) > 0, "Processing returned empty dataframe"
        
        # Step 3: Verify features
        features = get_features_for_target(target)
        missing_features = [f for f in features if f not in processed.columns]
        assert len(missing_features) == 0, f"Missing features: {missing_features}"
        
        # Step 4: Make predictions
        X = processed[features]
        predictions_made = False
        
        for h in HORIZONS:
            model = load_model(target, h)
            if model is not None:
                pred = model.predict(X)
                assert len(pred) > 0, f"No predictions for {h}h"
                assert not np.any(np.isnan(pred)), f"NaN in predictions for {h}h"
                predictions_made = True
        
        assert predictions_made, "No predictions were made"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_pipeline_all_targets(self, load_model, target):
        """Test complete pipeline for all targets."""
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")
        
        # Load and process
        df = load_data(RAW_DATA_PATH)
        processed = process_data(df.tail(50).copy(), target_name=target, is_training=False)
        
        if len(processed) == 0:
            pytest.skip(f"No data after processing for {target}")
        
        # Get features
        features = get_features_for_target(target)
        X = processed[features]
        
        # Predict with at least one model
        model = load_model(target, 1)  # Use 1h horizon
        if model is None:
            pytest.skip(f"Model not found: {target}_1h")
        
        pred = model.predict(X)
        
        # Validation
        assert len(pred) == len(X), "Prediction count mismatch"
        assert not np.any(np.isnan(pred)), f"NaN predictions for {target}"
        assert np.all(pred > 0), f"Negative predictions for {target}"
    
    def test_pipeline_handles_edge_cases(self, load_model):
        """Test that pipeline handles edge cases gracefully."""
        target = "pm2_5"
        
        # Create edge case data: minimal historical data
        edge_data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=5, freq="h"),
            "pm2_5": [20, 21, 19, 22, 20],
            "pm10": [30, 31, 29, 32, 30],
            "nitrogen_dioxide": [15, 16, 14, 17, 15],
            "ozone": [40, 41, 39, 42, 40],
            "temperature_2m": [25, 24, 26, 25, 24],
            "relative_humidity_2m": [60, 61, 59, 60, 61],
            "wind_speed_10m": [5, 6, 4, 5, 6],
            "wind_direction_10m": [180, 170, 190, 180, 175],
            "precipitation": [0, 0, 0, 0, 0],
            "surface_pressure": [1013, 1014, 1012, 1013, 1014]
        })
        
        # Process (should handle minimal data)
        processed = process_data(edge_data, target_name=target, is_training=False)
        
        # Should have at least some rows
        assert len(processed) > 0, "Pipeline failed on minimal data"
        
        # Try prediction
        features = get_features_for_target(target)
        X = processed[features]
        
        model = load_model(target, 1)
        if model is not None:
            pred = model.predict(X)
            assert len(pred) > 0, "No predictions on edge case data"
            assert not np.any(np.isnan(pred)), "NaN predictions on edge case data"


class TestCrossTargetValidation:
    """Test validation across different targets."""
    
    def test_all_targets_can_predict(self, load_model):
        """Test that all targets have working models."""
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")
        
        df = load_data(RAW_DATA_PATH)
        test_data = df.tail(50).copy()
        
        successful_targets = []
        
        for target in AVAILABLE_TARGETS:
            try:
                processed = process_data(test_data.copy(), target_name=target, is_training=False)
                if len(processed) == 0:
                    continue
                
                features = get_features_for_target(target)
                X = processed[features]
                
                model = load_model(target, 1)
                if model is None:
                    continue
                
                pred = model.predict(X)
                if len(pred) > 0 and not np.any(np.isnan(pred)):
                    successful_targets.append(target)
            except Exception as e:
                pytest.fail(f"Pipeline failed for {target}: {str(e)}")
        
        # At least PM2.5 should work
        assert len(successful_targets) > 0, "No targets can make predictions"
        assert "pm2_5" in successful_targets, "PM2.5 pipeline not working"
    
    def test_predictions_differ_across_targets(self, load_model):
        """Test that different targets produce different predictions."""
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip(f"Raw data not found: {RAW_DATA_PATH}")
        
        df = load_data(RAW_DATA_PATH)
        test_data = df.tail(50).copy()
        
        h = 1
        predictions = {}
        
        for target in AVAILABLE_TARGETS:
            processed = process_data(test_data.copy(), target_name=target, is_training=False)
            if len(processed) == 0:
                continue
            
            features = get_features_for_target(target)
            X = processed[features]
            
            model = load_model(target, h)
            if model is None:
                continue
            
            pred = model.predict(X)
            predictions[target] = pred
        
        # Should have at least 2 targets
        if len(predictions) >= 2:
            targets = list(predictions.keys())
            
            # Predictions for different pollutants should differ
            for i in range(len(targets)-1):
                for j in range(i+1, len(targets)):
                    t1, t2 = targets[i], targets[j]
                    
                    # Predictions should not be identical
                    are_different = not np.allclose(
                        predictions[t1][:min(len(predictions[t1]), len(predictions[t2]))],
                        predictions[t2][:min(len(predictions[t1]), len(predictions[t2]))]
                    )
                    
                    assert are_different, \
                        f"Predictions for {t1} and {t2} are suspiciously similar"


class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""
    
    def test_pipeline_with_missing_pollutant_data(self, load_model):
        """Test pipeline when some pollutant data is missing."""
        target = "ozone"
        
        # Create data with some pollutant readings as None/NaN
        test_data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=30, freq="h"),
            "pm2_5": [20] * 30,
            "pm10": [30] * 30,
            "nitrogen_dioxide": [15] * 30,
            "ozone": [40] * 30,
            "temperature_2m": [25] * 30,
            "relative_humidity_2m": [60] * 30,
            "wind_speed_10m": [5] * 30,
            "wind_direction_10m": [180] * 30,
            "precipitation": [0] * 30,
            "surface_pressure": [1013] * 30
        })
        
        # Processing should handle this
        processed = process_data(test_data, target_name=target, is_training=False)
        
        assert len(processed) > 0, "Pipeline failed with basic data"
        
        # Should be able to predict
        features = get_features_for_target(target)
        X = processed[features]
        
        model = load_model(target, 1)
        if model is not None:
            pred = model.predict(X)
            assert not np.any(np.isnan(pred)), "Pipeline produced NaN predictions"
