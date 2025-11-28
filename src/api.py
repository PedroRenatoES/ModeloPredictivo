import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict_manual import predict_manual

app = FastAPI(
    title="AirQuality_Pro API",
    description="API for predicting PM2.5 levels using XGBoost.",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    time: datetime
    pm2_5: float
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

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AirQuality_Pro"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert Pydantic model to dict
        data_dict = input_data.dict()
        
        # Run prediction logic
        predictions = predict_manual(data_dict)
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Prediction failed. Check server logs or model availability.")
            
        return {
            "input_time": input_data.time,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
