"""
Test suite for model regression testing.

Tests compare model metrics across horizons and against baseline models
to ensure model performance meets expectations.
"""
import pytest
import json
import os
import pandas as pd
import numpy as np
from src.config import MODELS_DIR, AVAILABLE_TARGETS, HORIZONS


class TestMetricsByHorizon:
    """Test model metrics across different horizons."""
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_metrics_file_exists(self, target):
        """Test that metrics file exists for each target."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        assert os.path.exists(metrics_path), \
            f"Metrics file not found: {metrics_path}"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_r2_score_positive(self, target):
        """Test that R² scores are positive (better than mean baseline)."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            r2 = metric.get("R2", None)
            horizon = metric.get("Horizon", "unknown")
            
            if r2 is not None:
                assert r2 > 0, \
                    f"{target} {horizon}: R²={r2:.3f} (should be >0)"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_r2_score_reasonable(self, target):
        """Test that R² scores are in reasonable range (>0.3 for good models)."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            r2 = metric.get("R2", None)
            horizon = metric.get("Horizon", "unknown")
            
            if r2 is not None:
                # Short horizons should have better R²
                if "1h" in horizon or "12h" in horizon:
                    assert r2 > 0.3, \
                        f"{target} {horizon}: R²={r2:.3f} is too low (expected >0.3 for short horizons)"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_mae_reasonable(self, target):
        """Test that MAE is reasonable relative to data scale."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Expected MAE ranges by target (rough estimates)
        max_mae_expected = {
            "pm2_5": 15.0,      # PM2.5 typically 0-100 µg/m³
            "pm10": 25.0,       # PM10 typically 0-150 µg/m³
            "ozone": 30.0,      # Ozone typically 0-200 µg/m³
            "nitrogen_dioxide": 20.0  # NO₂ typically 0-100 µg/m³
        }
        
        max_mae = max_mae_expected.get(target, 50.0)
        
        for metric in metrics:
            mae = metric.get("MAE", None)
            horizon = metric.get("Horizon", "unknown")
            
            if mae is not None:
                assert mae < max_mae, \
                    f"{target} {horizon}: MAE={mae:.3f} is too high (expected <{max_mae})"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_mape_reasonable(self, target):
        """Test that MAPE is reasonable (<50% for good models)."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            mape = metric.get("MAPE", None)
            horizon = metric.get("Horizon", "unknown")
            
            if mape is not None:
                # MAPE should be reasonable (<50% for useful predictions)
                assert mape < 50.0, \
                    f"{target} {horizon}: MAPE={mape:.1f}% is too high (expected <50%)"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_correlation_strong(self, target):
        """Test that correlation between predictions and actual is strong."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            corr = metric.get("Corr", None)
            horizon = metric.get("Horizon", "unknown")
            
            if corr is not None:
                # Correlation should be reasonably strong (>0.5)
                assert corr > 0.5, \
                    f"{target} {horizon}: Correlation={corr:.3f} is too low (expected >0.5)"


class TestBaselineComparison:
    """Test model performance against baseline (persistence model)."""
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_skill_score_positive(self, target):
        """Test that skill score is positive (model beats persistence baseline)."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            skill = metric.get("Skill", None)
            horizon = metric.get("Horizon", "unknown")
            
            if skill is not None:
                assert skill > 0, \
                    f"{target} {horizon}: Skill Score={skill:.2f}% (should be >0 to beat baseline)"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_model_better_than_baseline(self, target):
        """Test that model MAE is better than baseline MAE."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            mae = metric.get("MAE", None)
            base_mae = metric.get("Base_MAE", None)
            horizon = metric.get("Horizon", "unknown")
            
            if mae is not None and base_mae is not None:
                assert mae < base_mae, \
                    f"{target} {horizon}: MAE={mae:.3f} not better than baseline={base_mae:.3f}"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_short_horizon_high_skill(self, target):
        """Test that short horizons have high skill scores."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Short horizons (1h, 12h) should have skill >10%
        for metric in metrics:
            skill = metric.get("Skill", None)
            horizon = metric.get("Horizon", "unknown")
            
            if skill is not None and ("1h" in horizon or "12h" in horizon):
                assert skill > 10.0, \
                    f"{target} {horizon}: Skill={skill:.2f}% is too low for short horizon (expected >10%)"


class TestMetricConsistency:
    """Test consistency of metrics across horizons."""
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_all_horizons_have_metrics(self, target):
        """Test that all horizons have computed metrics."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Should have metrics for all horizons
        assert len(metrics) == len(HORIZONS), \
            f"{target}: Expected {len(HORIZONS)} horizons, found {len(metrics)}"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_metrics_complete(self, target):
        """Test that all required metrics are present."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        required_keys = ["MAE", "RMSE", "R2", "MAPE", "Corr", "Skill", "Base_MAE"]
        
        for metric in metrics:
            horizon = metric.get("Horizon", "unknown")
            for key in required_keys:
                assert key in metric, \
                    f"{target} {horizon}: Missing metric '{key}'"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_rmse_greater_than_mae(self, target):
        """Test that RMSE >= MAE (mathematical property)."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for metric in metrics:
            mae = metric.get("MAE", None)
            rmse = metric.get("RMSE", None)
            horizon = metric.get("Horizon", "unknown")
            
            if mae is not None and rmse is not None:
                assert rmse >= mae - 0.01, \
                    f"{target} {horizon}: RMSE={rmse:.3f} should be >= MAE={mae:.3f}"


class TestPerformanceDegradation:
    """Test that model performance degrades gracefully with horizon."""
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_mae_increases_with_horizon(self, target):
        """Test that MAE generally increases with forecast horizon."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Sort by horizon
        sorted_metrics = sorted(metrics, key=lambda x: int(x["Horizon"].split("_")[-1].replace("h", "")))
        
        # MAE should generally increase (allow some variance)
        maes = [m["MAE"] for m in sorted_metrics]
        
        # Check that longest horizon MAE is not better than shortest
        if len(maes) >= 2:
            assert maes[-1] >= maes[0] * 0.8, \
                f"{target}: Longest horizon MAE={maes[-1]:.3f} suspiciously better than shortest={maes[0]:.3f}"
    
    @pytest.mark.parametrize("target", AVAILABLE_TARGETS)
    def test_r2_decreases_with_horizon(self, target):
        """Test that R² generally decreases with forecast horizon."""
        metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Sort by horizon
        sorted_metrics = sorted(metrics, key=lambda x: int(x["Horizon"].split("_")[-1].replace("h", "")))
        
        # R² should generally decrease
        r2s = [m["R2"] for m in sorted_metrics]
        
        # Check that longest horizon R² is not much better than shortest
        if len(r2s) >= 2:
            assert r2s[-1] <= r2s[0] * 1.2, \
                f"{target}: Longest horizon R²={r2s[-1]:.3f} suspiciously better than shortest={r2s[0]:.3f}"
