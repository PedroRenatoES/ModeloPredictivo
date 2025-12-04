import sys
import os
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import uvicorn
import pandas as pd
import xgboost as xgb
import numpy as np
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODELS_DIR, AVAILABLE_TARGETS, HORIZONS, get_features_for_target
from src.data_processing import process_data

app = FastAPI(
    title="API de Predicción de Calidad del Aire Multi-Target",
    description="API para predecir múltiples contaminantes atmosféricos (PM2.5, PM10, Ozono, NO₂) usando XGBoost.",
    version="2.0.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

class PredictionInput(BaseModel):
    """Input data for making predictions"""
    time: datetime
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    nitrogen_dioxide: Optional[float] = None
    ozone: Optional[float] = None
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

class RiskResponse(BaseModel):
    """Response containing prediction and risk classification"""
    target: str
    predicted_value: float
    risk_level: str
    unit: str
    message: str

@app.get("/health")
def health_check():
    """Endpoint de verificación de estado del servicio"""
    return {
        "estado": "ok", 
        "servicio": "API de Predicción de Calidad del Aire Multi-Target",
        "targets_disponibles": AVAILABLE_TARGETS,
        "horizontes_disponibles": HORIZONS
    }

@app.get("/targets")
def get_available_targets():
    """Obtener lista de contaminantes disponibles para predicción"""
    return {
        "targets_disponibles": AVAILABLE_TARGETS,
        "descripciones": {
            "pm2_5": "Material Particulado Fino (≤2.5 μm)",
            "pm10": "Material Particulado Grueso (≤10 μm)",
            "ozone": "Ozono Troposférico (O₃)",
            "nitrogen_dioxide": "Dióxido de Nitrógeno (NO₂)"
        }
    }

@app.post("/predict/{target}")
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
                detail=f"Target inválido. Debe ser uno de: {AVAILABLE_TARGETS}"
            )
        
        if not input_data:
            raise HTTPException(status_code=400, detail="La lista de datos de entrada no puede estar vacía")

        # Parse horizons
        if horizons:
            try:
                horizon_list = [int(h.strip()) for h in horizons.split(',')]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Los horizontes deben ser enteros separados por comas (ej: '1,12,24')"
                )
        else:
            horizon_list = HORIZONS
        
        # Convert input to DataFrame
        input_list = [item.model_dump() for item in input_data]
        df = pd.DataFrame(input_list)
        
        # Convert pollutant columns to float (handles None/null properly)
        pollutant_cols = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone']
        for col in pollutant_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
                detail=f"Características faltantes después del procesamiento: {missing_features}"
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
                    "valor": None,
                    "error": "Modelo no encontrado. Por favor entrene el modelo primero.",
                    "tiempo_predicho": None
                }
                continue
            
            # Load model
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            
            # Predict
            pred_value = float(model.predict(X)[0])
            pred_time = current_time + pd.Timedelta(hours=h)
            
            predictions[f"{h}h"] = {
                "valor": round(pred_value, 2),
                "tiempo_predicho": pred_time.isoformat(),
                "horas_horizonte": h
            }
        
        # Determine unit based on target
        units = {
            "pm2_5": "μg/m³",
            "pm10": "μg/m³",
            "ozone": "μg/m³",
            "nitrogen_dioxide": "μg/m³"
        }
        
        return {
            "target": target,
            "tiempo_entrada": current_time.isoformat(),
            "predicciones": predictions,
            "unidad": units.get(target, "μg/m³")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/metrics/{target}/r2")
def get_average_r2(target: str):
    """
    Obtener el puntaje R2 promedio de los modelos entrenados para un target específico.
    """
    if target not in AVAILABLE_TARGETS:
        raise HTTPException(status_code=400, detail=f"Target inválido. Debe ser uno de: {AVAILABLE_TARGETS}")
    
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{target}.json")
    
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail=f"Métricas no encontradas para {target}. Por favor entrene el modelo primero.")
        
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        if not metrics:
             raise HTTPException(status_code=404, detail="No hay datos de métricas disponibles.")

        r2_scores = [m["R2"] for m in metrics]
        avg_r2 = sum(r2_scores) / len(r2_scores)
        
        return {
            "target": target,
            "r2_promedio": round(avg_r2, 4),
            "cantidad_modelos": len(metrics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer métricas: {str(e)}")

@app.post("/predict/{target}/5-horizons")
def predict_5_horizons(
    target: str,
    input_data: List[PredictionInput] = Body(..., description="Historical data")
):
    """
    Realizar predicciones para 5 horizontes específicos: 1, 12, 24, 72, 168 horas.
    """
    # Force the 5 horizons
    five_horizons = "1,12,24,72,168"
    return predict_target(target=target, input_data=input_data, horizons=five_horizons)

@app.post("/predict/{target}/risk")
def predict_risk(
    target: str,
    input_data: List[PredictionInput] = Body(..., description="Historical data")
):
    """
    Predecir el valor a 1 hora y clasificar el nivel de riesgo.
    Niveles de Riesgo (Ejemplo para PM2.5/PM10):
    - Bajo: < 35
    - Medio: 35 - 55
    - Alto: > 55
    """
    # Get 1h prediction
    pred_response = predict_target(target=target, input_data=input_data, horizons="1")
    
    # Extract value
    try:
        pred_value = pred_response["predicciones"]["1h"]["valor"]
    except KeyError:
         raise HTTPException(status_code=500, detail="No se pudo generar la predicción a 1h para evaluación de riesgo")
         
    if pred_value is None:
        raise HTTPException(status_code=500, detail="El modelo retornó None para la predicción")

    # Classify Risk (Simple Switch/If-Else)
    # Note: These thresholds are examples. Real thresholds depend on the pollutant and regulations.
    if pred_value < 35:
        risk = "Bajo"
        msg = "La calidad del aire es buena. Disfrute de sus actividades al aire libre."
    elif 35 <= pred_value < 55:
        risk = "Medio"
        msg = "Los grupos sensibles pueden experimentar efectos en la salud."
    else: # >= 55
        risk = "Alto"
        msg = "Todos pueden comenzar a experimentar efectos en la salud; los grupos sensibles pueden experimentar efectos más graves."
        
    return {
        "target": target,
        "valor_predicho": pred_value,
        "nivel_riesgo": risk,
        "unidad": pred_response["unidad"],
        "mensaje": msg
    }

def calculate_pollutant_ica(pollutant: str, value: float) -> int:
    """
    Calcula el ICA para un contaminante específico según la Orden TEC/351/2019.
    
    Args:
        pollutant: Nombre del contaminante (pm2_5, pm10, ozone, nitrogen_dioxide)
        value: Concentración del contaminante en μg/m³
    
    Returns:
        int: Índice ICA de 1 (Buena) a 6 (Extremadamente desfavorable)
    
    Rangos basados en la regulación española (Orden TEC/351/2019):
    - 1: Buena
    - 2: Razonablemente buena  
    - 3: Regular
    - 4: Desfavorable
    - 5: Muy desfavorable
    - 6: Extremadamente desfavorable
    """
    # Thresholds según la Orden TEC/351/2019
    thresholds = {
        "pm2_5": [10, 20, 25, 50, 75, float('inf')],  # μg/m³
        "pm10": [20, 40, 50, 100, 150, float('inf')],  # μg/m³
        "ozone": [50, 100, 130, 240, 380, float('inf')],  # μg/m³
        "nitrogen_dioxide": [40, 90, 120, 230, 340, float('inf')]  # μg/m³
    }
    
    if pollutant not in thresholds:
        return 1  # Default to "Buena" if pollutant not recognized
    
    limits = thresholds[pollutant]
    
    # Determinar categoría ICA
    for ica_level, threshold in enumerate(limits, start=1):
        if value < threshold:
            return ica_level
    
    return 6  # Extremadamente desfavorable

@app.post("/predict/ica")
def predict_ica(
    input_data: List[PredictionInput] = Body(
        ...,
        description="List of historical data points (last 24h recommended) ending with current conditions"
    )
):
    """
    Calcula el Índice de Calidad del Aire (ICA) usando predicciones a 1 hora
    de todos los contaminantes disponibles (PM2.5, PM10, Ozono, NO₂).
    
    El ICA se calcula según la Orden TEC/351/2019 española, tomando el peor
    valor entre todos los contaminantes medidos.
    
    Args:
        input_data: Lista de datos históricos (últimas 24h recomendadas) terminando con condiciones actuales
    
    Returns:
        int: Valor numérico del ICA de 1 a 6
        - 1: Buena
        - 2: Razonablemente buena
        - 3: Regular
        - 4: Desfavorable
        - 5: Muy desfavorable
        - 6: Extremadamente desfavorable
    """
    try:
        if not input_data:
            raise HTTPException(status_code=400, detail="La lista de datos de entrada no puede estar vacía")
        
        # Obtener predicciones a 1 hora para cada contaminante
        pollutants_to_check = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"]
        ica_values = []
        
        for pollutant in pollutants_to_check:
            try:
                # Obtener predicción a 1 hora
                pred_response = predict_target(target=pollutant, input_data=input_data, horizons="1")
                
                # Extraer el valor predicho
                pred_value = pred_response["predicciones"]["1h"]["valor"]
                
                if pred_value is not None:
                    # Calcular ICA para este contaminante
                    ica = calculate_pollutant_ica(pollutant, pred_value)
                    ica_values.append(ica)
                    
            except Exception as e:
                # Si un modelo no está disponible, continuamos con los demás
                continue
        
        if not ica_values:
            raise HTTPException(
                status_code=500,
                detail="No se pudo calcular el ICA. Asegúrese de que al menos un modelo esté entrenado."
            )
        
        # El ICA final es el peor valor (máximo) entre todos los contaminantes
        final_ica = max(ica_values)
        
        return final_ica
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al calcular ICA: {str(e)}")

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
    Endpoint de predicción retrocompatible.
    Por defecto predice PM2.5.
    """
    return predict_target(target=target, input_data=input_data, horizons=horizons)

if __name__ == "__main__":
    print("Starting AirQuality Multi-Target Prediction API...")
    print(f"Available targets: {AVAILABLE_TARGETS}")
    print(f"Available horizons: {HORIZONS}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

