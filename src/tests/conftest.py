"""
Pytest configuration and shared fixtures for test suite.
"""
import pytest
import sys
import os
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    MODELS_DIR,
    AVAILABLE_TARGETS,
    HORIZONS,
    get_features_for_target
)


@pytest.fixture(scope="session")
def raw_data():
    """Load raw data for testing."""
    if not os.path.exists(RAW_DATA_PATH):
        pytest.skip(f"Raw data not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])


@pytest.fixture(scope="session")
def processed_data():
    """Load processed data for testing."""
    data_path = PROCESSED_DATA_PATH.replace("train_data.csv", "train_data_pm2_5.csv")
    if not os.path.exists(data_path):
        pytest.skip(f"Processed data not found at {data_path}")
    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def sample_input_data():
    """Create sample input data for predictions."""
    return pd.DataFrame([{
        "time": "2025-07-01T12:00:00",
        "pm2_5": 15.5,
        "pm10": 25.0,
        "nitrogen_dioxide": 20.0,
        "ozone": 45.0,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 5.5,
        "wind_direction_10m": 180.0,
        "precipitation": 0.0,
        "surface_pressure": 1013.0
    }])


@pytest.fixture(params=AVAILABLE_TARGETS, scope="session")
def target_name(request):
    """Parametrize tests across all available targets."""
    return request.param


@pytest.fixture(params=HORIZONS, scope="session")
def horizon(request):
    """Parametrize tests across all horizons."""
    return request.param


@pytest.fixture(scope="session")
def models_cache():
    """Cache for loaded models to avoid reloading."""
    return {}


@pytest.fixture
def load_model(models_cache):
    """Load a specific model with caching."""
    def _load_model(target_name, horizon):
        key = f"{target_name}_{horizon}h"
        if key in models_cache:
            return models_cache[key]
        
        model_path = os.path.join(MODELS_DIR, f"xgboost_{target_name}_{horizon}h.json")
        if not os.path.exists(model_path):
            return None
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models_cache[key] = model
        return model
    
    return _load_model


@pytest.fixture(scope="session")
def features_by_target():
    """Get features for each target."""
    return {target: get_features_for_target(target) for target in AVAILABLE_TARGETS}
