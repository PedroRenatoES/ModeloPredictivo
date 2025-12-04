"""
Test suite for model inference performance.

Tests verify that inference time meets the <1 second requirement
for real-time predictions.
"""
import pytest
import time
import pandas as pd
import numpy as np
from src.config import AVAILABLE_TARGETS, HORIZONS, get_features_for_target


class TestInferencePerformance:
    """Test model inference performance requirements."""
    
    def test_single_prediction_under_1_second(self, load_model, sample_input_data):
        """Test that single prediction completes in <1 second."""
        target = "pm2_5"
        horizon = 1
        
        model = load_model(target, horizon)
        if model is None:
            pytest.skip(f"Model for {target}_{horizon}h not found")
        
        # Prepare input data
        from src.data_processing import process_data
        processed = process_data(sample_input_data.copy(), target_name=target, is_training=False)
        features = get_features_for_target(target)
        X = processed[features]
        
        # Measure inference time
        start_time = time.time()
        prediction = model.predict(X)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        assert inference_time < 1.0, \
            f"Inference too slow: {inference_time:.3f}s (requirement: <1s)"
        
        # Also check prediction is reasonable
        assert len(prediction) == 1, "Should return single prediction"
        assert prediction[0] > 0, "Prediction should be positive"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_all_models_performance(self, load_model, sample_input_data, target, horizon):
        """Test inference performance for all target-horizon combinations."""
        model = load_model(target, horizon)
        if model is None:
            pytest.skip(f"Model for {target}_{horizon}h not found")
        
        from src.data_processing import process_data
        processed = process_data(sample_input_data.copy(), target_name=target, is_training=False)
        features = get_features_for_target(target)
        X = processed[features]
        
        start_time = time.time()
        prediction = model.predict(X)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        assert inference_time < 1.0, \
            f"{target}_{horizon}h inference too slow: {inference_time:.3f}s"
    
    def test_batch_prediction_performance(self, load_model):
        """Test batch prediction with multiple samples."""
        target = "pm2_5"
        horizon = 1
        
        model = load_model(target, horizon)
        if model is None:
            pytest.skip(f"Model for {target}_{horizon}h not found")
        
        # Create batch of 10 samples
        n_samples = 10
        batch_data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
            "pm2_5": np.random.uniform(10, 50, n_samples),
            "pm10": np.random.uniform(20, 80, n_samples),
            "nitrogen_dioxide": np.random.uniform(10, 60, n_samples),
            "ozone": np.random.uniform(30, 100, n_samples),
            "temperature_2m": np.random.uniform(15, 30, n_samples),
            "relative_humidity_2m": np.random.uniform(40, 80, n_samples),
            "wind_speed_10m": np.random.uniform(0, 15, n_samples),
            "wind_direction_10m": np.random.uniform(0, 360, n_samples),
            "precipitation": np.random.uniform(0, 5, n_samples),
            "surface_pressure": np.random.uniform(1000, 1020, n_samples)
        })
        
        from src.data_processing import process_data
        processed = process_data(batch_data, target_name=target, is_training=False)
        features = get_features_for_target(target)
        X = processed[features]
        
        start_time = time.time()
        predictions = model.predict(X)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_sample = total_time / len(predictions)
        
        # Total batch should be under 1 second
        assert total_time < 1.0, \
            f"Batch prediction too slow: {total_time:.3f}s for {n_samples} samples"
        
        # Average per sample should be well under 1 second
        assert avg_time_per_sample < 0.1, \
            f"Average prediction too slow: {avg_time_per_sample:.3f}s per sample"
    
    def test_multi_horizon_prediction_performance(self, load_model, sample_input_data):
        """Test performance when predicting all horizons for one target."""
        target = "pm2_5"
        
        from src.data_processing import process_data
        processed = process_data(sample_input_data.copy(), target_name=target, is_training=False)
        features = get_features_for_target(target)
        X = processed[features]
        
        start_time = time.time()
        
        # Predict for all horizons
        predictions = {}
        for h in HORIZONS:
            model = load_model(target, h)
            if model is not None:
                predictions[h] = model.predict(X)[0]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Predicting all 5 horizons should be under 1 second total
        assert total_time < 1.0, \
            f"Multi-horizon prediction too slow: {total_time:.3f}s for {len(HORIZONS)} horizons"
        
        # Should have predictions for all horizons
        assert len(predictions) == len(HORIZONS), "Missing predictions for some horizons"


class TestModelLoadingPerformance:
    """Test model loading performance."""
    
    def test_model_loading_time(self):
        """Test that model loading is reasonably fast."""
        import xgboost as xgb
        import os
        from src.config import MODELS_DIR
        
        target = "pm2_5"
        horizon = 1
        model_path = os.path.join(MODELS_DIR, f"xgboost_{target}_{horizon}h.json")
        
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")
        
        start_time = time.time()
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        end_time = time.time()
        
        load_time = end_time - start_time
        
        # Model loading should be fast (<2 seconds)
        assert load_time < 2.0, \
            f"Model loading too slow: {load_time:.3f}s"
