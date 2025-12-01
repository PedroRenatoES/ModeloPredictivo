import sys
import os
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import uvicorn
import pandas as pd
import xgboost as xgb
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODELS_DIR, AVAILABLE_TARGETS, HORIZONS, get_features_for_target
from src.data_processing import process_data

app = FastAPI(
    title="AirQuality Multi-Target Prediction API",
    description="API for predicting multiple air quality pollutants (PM2.5, PM10, Ozone, NO₂) using XGBoost.",
    version="2.0.0"
)

class PredictionInput(BaseModel):
    """Input data for making predictions"""
    time: datetime
    pm2_5: float
    pm10: float
    nitrogen_dioxide: float
    ozone: float
    temperature_2m: float
    relative_humidity_2m: float
    wind_speed_10m: float
    wind_direction_10m: float
    precipitation: float
    surface_pressure: float

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }

class PredictionResponse(BaseModel):
    """Response containing predictions for multiple horizons"""
    target: str
    input_time: datetime
    predictions: dict
    unit: str

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "service": "AirQuality Multi-Target API",
        "available_targets": AVAILABLE_TARGETS,
        "available_horizons": HORIZONS
    }

@app.get("/targets")
def get_available_targets():
    """Get list of available prediction targets"""
    return {
        "available_targets": AVAILABLE_TARGETS,
        "descriptions": {
            "pm2_5": "Fine Particulate Matter (≤2.5 μm)",
            "pm10": "Coarse Particulate Matter (≤10 μm)",
            "ozone": "Ground-level Ozone (O₃)",
            "nitrogen_dioxide": "Nitrogen Dioxide (NO₂)"
        }
    }

@app.post("/predict/{target}", response_model=PredictionResponse)
def predict_target(
    target: str,
    input_data: List[PredictionInput] = Body(
        ...,
        example=[
            {
                "time": "2025-07-01T11:00:00",
                "pm2_5": 14.2,
                "pm10": 22.1,
                "nitrogen_dioxide": 18.5,
                "ozone": 40.2,
                "temperature_2m": 24.5,
                "relative_humidity_2m": 62.0,
                "wind_speed_10m": 5.2,
                "wind_direction_10m": 175.0,
                "precipitation": 0.0,
                "surface_pressure": 1012.5
            },
            {
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
            }
        ],
        description="List of historical data points (last 24h recommended) ending with current conditions"
    ),
    horizons: Optional[str] = Query(None, description="Comma-separated horizons (e.g., '1,12,24')")
):
    """
    Predict air quality for a specific target variable.
    
    Args:
        target: Target pollutant to predict (pm2_5, pm10, ozone, nitrogen_dioxide)
        input_data: List of historical data points (last 24h recommended) ending with current conditions
        horizons: Optional comma-separated list of forecast horizons in hours
    
    Returns:
        Predictions for the specified target at different time horizons
    """
    try:
        # Validate target
        if target not in AVAILABLE_TARGETS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid target. Must be one of: {AVAILABLE_TARGETS}"
            )
        
        if not input_data:
            raise HTTPException(status_code=400, detail="Input list cannot be empty")

        # Parse horizons
        if horizons:
            try:
                horizon_list = [int(h.strip()) for h in horizons.split(',')]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Horizons must be comma-separated integers (e.g., '1,12,24')"
                )
        else:
            horizon_list = HORIZONS
        
        # Convert input to DataFrame
        input_list = [item.model_dump() for item in input_data]
        df = pd.DataFrame(input_list)
        
        # Ensure sorted by time
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Process features (without creating future targets since we're predicting)
        # Passing the full history allows calculating rolling stats correctly
        df_processed = process_data(df, target_name=target, is_training=False)
        
        # We only care about predicting for the LAST timestamp provided (the "current" moment)
        current_data_processed = df_processed.iloc[[-1]]
        
        # Get required features for this target
        required_features = get_features_for_target(target)
        
        # Verify all features are present
        missing_features = [f for f in required_features if f not in current_data_processed.columns]
        if missing_features:
            raise HTTPException(
                status_code=500,
                detail=f"Missing features after processing: {missing_features}"
            )
        
        X = current_data_processed[required_features]
        
        # Load models and make predictions
        predictions = {}
        
        # Get the time from the last data point
        current_time = df.iloc[-1]['time']
        
        for h in horizon_list:
            model_path = os.path.join(MODELS_DIR, f"xgboost_{target}_{h}h.json")
            
            if not os.path.exists(model_path):
                predictions[f"{h}h"] = {
                    "value": None,
                    "error": "Model not found. Please train model first.",
                    "predicted_time": None
                }
                continue
            
            # Load model
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            
            # Predict
            pred_value = float(model.predict(X)[0])
            pred_time = current_time + pd.Timedelta(hours=h)
            
            predictions[f"{h}h"] = {
                "value": round(pred_value, 2),
                "predicted_time": pred_time.isoformat(),
                "horizon_hours": h
            }
        
        # Determine unit based on target
        units = {
            "pm2_5": "μg/m³",
            "pm10": "μg/m³",
            "ozone": "μg/m³",
            "nitrogen_dioxide": "μg/m³"
        }
        
        return PredictionResponse(
            target=target,
            input_time=current_time,
            predictions=predictions,
            unit=units.get(target, "μg/m³")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict")
def predict_default(
    input_data: List[PredictionInput] = Body(
        ...,
        example=[
            {
                "time": "2025-07-01T11:00:00",
                "pm2_5": 14.2,
                "pm10": 22.1,
                "nitrogen_dioxide": 18.5,
                "ozone": 40.2,
                "temperature_2m": 24.5,
                "relative_humidity_2m": 62.0,
                "wind_speed_10m": 5.2,
                "wind_direction_10m": 175.0,
                "precipitation": 0.0,
                "surface_pressure": 1012.5
            },
            {
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
            }
        ],
        description="List of historical data points (last 24h recommended) ending with current conditions"
    ),
    target: str = Query("pm2_5", description="Target variable to predict"),
    horizons: Optional[str] = Query(None, description="Comma-separated horizons")
):
    """
    Backward-compatible prediction endpoint.
    Defaults to PM2.5 prediction.
    """
    return predict_target(target=target, input_data=input_data, horizons=horizons)

if __name__ == "__main__":
    print("Starting AirQuality Multi-Target Prediction API...")
    print(f"Available targets: {AVAILABLE_TARGETS}")
    print(f"Available horizons: {HORIZONS}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

