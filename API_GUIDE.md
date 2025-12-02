# API Multi-Target - Guía Completa de Uso

## 🚀 Iniciar el Servidor

```bash
# Desde el directorio raíz del proyecto
python src/api.py
```

O usando uvicorn directamente:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

El servidor iniciará en: **http://localhost:8000**

Documentación interactiva: **http://localhost:8000/docs**

---

## 📡 Endpoints Disponibles

### 1. Health Check
```http
GET /health
```

**Descripción:** Verifica el estado del servicio y retorna información sobre targets y horizontes disponibles.

**Respuesta:**
```json
{
  "status": "ok",
  "service": "AirQuality Multi-Target API",
  "available_targets": ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"],
  "available_horizons": [1, 12, 24, 72, 168]
}
```

---

### 2. Listar Targets Disponibles
```http
GET /targets
```

**Descripción:** Obtiene la lista de contaminantes que el sistema puede predecir con sus descripciones.

**Respuesta:**
```json
{
  "available_targets": ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"],
  "descriptions": {
    "pm2_5": "Fine Particulate Matter (≤2.5 μm)",
    "pm10": "Coarse Particulate Matter (≤10 μm)",
    "ozone": "Ground-level Ozone (O₃)",
    "nitrogen_dioxide": "Nitrogen Dioxide (NO₂)"
  }
}
```

---

### 3. Obtener Métricas R² de un Target
```http
GET /metrics/{target}/r2
```

**Descripción:** Retorna el R² promedio de todos los modelos entrenados para un contaminante específico.

**Parámetros:**
- `target` (path): pm2_5, pm10, ozone, nitrogen_dioxide

**Ejemplo:**
```bash
curl http://localhost:8000/metrics/pm2_5/r2
```

**Respuesta:**
```json
{
  "target": "pm2_5",
  "average_r2": 0.7833,
  "models_count": 5
}
```

**Campos de la respuesta:**
- `target`: El contaminante consultado
- `average_r2`: Promedio del coeficiente de determinación R² de todos los horizontes
- `models_count`: Cantidad de modelos (horizontes) incluidos en el promedio

**Nota:** Requiere haber entrenado los modelos primero con `python main.py --target {target}`

---

### 4. Predecir con 5 Horizontes Fijos
```http
POST /predict/{target}/5-horizons
```

**Descripción:** Realiza predicciones para exactamente 5 horizontes de tiempo predefinidos: 1h, 12h, 24h, 72h y 168h.

**Parámetros:**
- `target` (path): pm2_5, pm10, ozone, nitrogen_dioxide

**Body (JSON) - Lista de datos históricos:**
```json
[
  {
    "time": "2025-11-30T17:00:00",
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
    "time": "2025-11-30T18:00:00",
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
]
```

**Respuesta:**
```json
{
  "target": "pm2_5",
  "input_time": "2025-11-30T18:00:00",
  "predictions": {
    "1h": {
      "value": 30.2,
      "predicted_time": "2025-11-30T19:00:00",
      "horizon_hours": 1
    },
    "12h": {
      "value": 60.87,
      "predicted_time": "2025-12-01T06:00:00",
      "horizon_hours": 12
    },
    "24h": {
      "value": 41.7,
      "predicted_time": "2025-12-01T18:00:00",
      "horizon_hours": 24
    },
    "72h": {
      "value": 19.63,
      "predicted_time": "2025-12-03T18:00:00",
      "horizon_hours": 72
    },
    "168h": {
      "value": 14.34,
      "predicted_time": "2025-12-07T18:00:00",
      "horizon_hours": 168
    }
  },
  "unit": "μg/m³"
}
```

**Campos de la respuesta:**
- `target`: Contaminante predicho
- `input_time`: Momento desde el cual se hace la predicción (último dato enviado)
- `predictions`: Diccionario con predicciones para cada horizonte
  - Clave: horizonte (ej: "1h", "12h")
  - `value`: Valor predicho del contaminante
  - `predicted_time`: Momento futuro al que corresponde la predicción
  - `horizon_hours`: Cantidad de horas en el futuro
- `unit`: Unidad de medida (μg/m³)

---

### 5. Clasificación de Riesgo
```http
POST /predict/{target}/risk
```

**Descripción:** Predice el valor a 1 hora y clasifica el nivel de riesgo en base a umbrales predefinidos.

**Parámetros:**
- `target` (path): pm2_5, pm10, ozone, nitrogen_dioxide

**Body (JSON):** Mismo formato que el endpoint anterior

**Respuesta:**
```json
{
  "target": "pm2_5",
  "predicted_value": 30.2,
  "risk_level": "Low",
  "unit": "μg/m³",
  "message": "Air quality is good. Enjoy your outdoor activities."
}
```

**Campos de la respuesta:**
- `target`: Contaminante evaluado
- `predicted_value`: Valor predicho a 1 hora
- `risk_level`: Nivel de riesgo ("Low", "Medium", "High")
- `unit`: Unidad de medida
- `message`: Mensaje descriptivo sobre la calidad del aire

**Umbrales de clasificación:**
- **Low (Bajo)**: < 35 μg/m³ - Calidad del aire buena
- **Medium (Medio)**: 35-55 μg/m³ - Grupos sensibles pueden verse afectados
- **High (Alto)**: ≥ 55 μg/m³ - Toda la población puede experimentar efectos

---

### 6. Predecir para un Target Específico (Horizontes Personalizados)
```http
POST /predict/{target}?horizons=1,12,24
```

**Descripción:** Permite predecir con horizontes de tiempo personalizados.

**Parámetros:**
- `target` (path): pm2_5, pm10, ozone, nitrogen_dioxide
- `horizons` (query, opcional): horizontes separados por comas (ej: "1,12,24"). Si se omite, usa todos los horizontes disponibles.

**Body (JSON):** Lista de datos históricos (igual que el endpoint 4)

**Respuesta:** Mismo formato que el endpoint 4, pero solo con los horizontes solicitados

---

### 7. Endpoint Compatible con Versión Anterior
```http
POST /predict?target=pm2_5&horizons=1,24
```

**Descripción:** Endpoint retrocompatible que por defecto predice PM2.5.

**Body:** Mismo formato que anteriores

---

## 🔍 Cómo Funcionan las APIs

### Arquitectura de la API

La API está construida con **FastAPI** y sigue una arquitectura RESTful con los siguientes componentes:

1. **FastAPI Application** (`src/api.py`)
   - Define los endpoints y sus rutas
   - Valida inputs usando Pydantic
   - Maneja errores y excepciones
   - Sirve documentación automática

2. **Modelos de Datos** (Pydantic Models)
   - `PredictionInput`: Valida los datos de entrada
   - `PredictionResponse`: Estructura la respuesta de predicciones
   - `RiskResponse`: Estructura la respuesta de clasificación de riesgo

3. **Procesamiento de Datos** (`src/data_processing.py`)
   - Transforma datos crudos en features
   - Calcula rolling statistics y lags
   - Crea features temporales (sin, cos)

4. **Modelos ML** (XGBoost)
   - Modelos entrenados previamente
   - Un modelo por cada combinación target-horizonte
   - Guardados como archivos JSON en `models/`

### Flujo de una Predicción

```
1. Cliente envía POST request
   ↓
2. FastAPI valida el JSON con Pydantic
   ↓
3. Se convierte la lista de inputs a DataFrame
   ↓
4. process_data() calcula features (lags, rolling stats, etc.)
   ↓
5. Se selecciona el último timestamp (momento actual)
   ↓
6. Para cada horizonte solicitado:
   - Se carga el modelo correspondiente
   - Se hace la predicción
   - Se calcula el tiempo futuro
   ↓
7. Se estructura la respuesta en formato JSON
   ↓
8. FastAPI retorna la respuesta al cliente
```

### ¿Por Qué Enviar Múltiples Datos Históricos?

Los modelos XGBoost necesitan **features derivadas** que se calculan a partir de la historia:

- **Lags**: Valor del contaminante hace 1h y 24h
- **Rolling Statistics**: Media y desviación estándar de las últimas 24h
- **Componentes de viento**: Transformación de velocidad/dirección a componentes U/V

Enviar solo el dato actual NO permitiría calcular estas features correctamente.

**Recomendación:** Enviar las últimas 24 horas de datos para asegurar cálculos precisos.

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Obtener R² Promedio

```python
import requests

response = requests.get("http://localhost:8000/metrics/pm2_5/r2")
data = response.json()

print(f"R² promedio de PM2.5: {data['average_r2']:.4f}")
print(f"Basado en {data['models_count']} modelos")
```

### Ejemplo 2: Predicción con 5 Horizontes

```python
import requests
from datetime import datetime, timedelta

# Generar datos históricos (últimas 24h)
historical_data = []
for i in range(24, 0, -1):
    historical_data.append({
        "time": (datetime.now() - timedelta(hours=i)).isoformat(),
        "pm2_5": 15.0 + i * 0.5,  # Valores de ejemplo
        "pm10": 25.0,
        "nitrogen_dioxide": 20.0,
        "ozone": 45.0,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 5.5,
        "wind_direction_10m": 180.0,
        "precipitation": 0.0,
        "surface_pressure": 1013.0
    })

response = requests.post(
    "http://localhost:8000/predict/pm2_5/5-horizons",
    json=historical_data
)

predictions = response.json()
print(f"Predicciones para {predictions['target']}:")
for horizon, pred in predictions['predictions'].items():
    print(f"  {horizon}: {pred['value']} μg/m³ a las {pred['predicted_time']}")
```

### Ejemplo 3: Sistema de Alertas con Clasificación de Riesgo

```python
import requests

def check_air_quality_risk(historical_data, target="pm2_5"):
    """
    Verifica el nivel de riesgo de calidad del aire
    """
    response = requests.post(
        f"http://localhost:8000/predict/{target}/risk",
        json=historical_data
    )
    
    result = response.json()
    
    print(f"🌡️  Predicción a 1 hora: {result['predicted_value']} {result['unit']}")
    print(f"⚠️  Nivel de riesgo: {result['risk_level']}")
    print(f"💬 {result['message']}")
    
    # Enviar alerta si es riesgo alto
    if result['risk_level'] == "High":
        send_alert(result)
    
    return result

def send_alert(risk_data):
    """Envía alerta por email, SMS, etc."""
    print(f"🚨 ALERTA: Riesgo alto de {risk_data['target']}!")
```

### Ejemplo 4: Dashboard Multi-Contaminante

```python
import requests

def get_all_pollutants_forecast(historical_data):
    """
    Obtiene predicciones para todos los contaminantes
    """
    targets = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"]
    forecasts = {}
    
    for target in targets:
        response = requests.post(
            f"http://localhost:8000/predict/{target}/5-horizons",
            json=historical_data
        )
        forecasts[target] = response.json()
    
    return forecasts

def display_dashboard(forecasts):
    """
    Muestra un dashboard simple en consola
    """
    for target, data in forecasts.items():
        print(f"\n📊 {target.upper()}")
        print(f"   Unidad: {data['unit']}")
        for horizon, pred in data['predictions'].items():
            print(f"   {horizon:>5}: {pred['value']:>6.2f} μg/m³")
```

---

## 🔄 Casos de Uso Comunes

### Caso 1: Integración con Sistema de Monitoreo

```python
import requests
import time
from datetime import datetime

def continuous_monitoring():
    """
    Monitoreo continuo que actualiza predicciones cada hora
    """
    while True:
        # Obtener datos actuales de sensores
        current_data = get_sensor_data()  # Tu función
        
        # Hacer predicción
        response = requests.post(
            "http://localhost:8000/predict/pm2_5/risk",
            json=current_data
        )
        
        risk = response.json()
        log_prediction(risk)  # Guardar en base de datos
        
        if risk['risk_level'] != 'Low':
            notify_users(risk)
        
        time.sleep(3600)  # Esperar 1 hora
```

### Caso 2: API Gateway Pattern

```python
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

AIR_QUALITY_API = "http://localhost:8000"

@app.get("/dashboard/{location}")
async def get_location_dashboard(location: str):
    """
    Endpoint que combina múltiples llamadas a la API de predicción
    """
    # Obtener datos históricos de la ubicación
    historical = get_location_data(location)
    
    # Obtener predicciones
    predictions = requests.post(
        f"{AIR_QUALITY_API}/predict/pm2_5/5-horizons",
        json=historical
    ).json()
    
    # Obtener métricas del modelo
    metrics = requests.get(
        f"{AIR_QUALITY_API}/metrics/pm2_5/r2"
    ).json()
    
    return {
        "location": location,
        "predictions": predictions,
        "model_accuracy": metrics['average_r2'],
        "timestamp": datetime.now().isoformat()
    }
```

---

## ⚠️ Manejo de Errores

### Error 400: Target Inválido
```json
{
  "detail": "Invalid target. Must be one of: ['pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide']"
}
```

### Error 404: Métricas No Encontradas
```json
{
  "detail": "Metrics not found for pm2_5. Please train the model first."
}
```
**Solución:** Entrenar los modelos
```bash
python main.py --target pm2_5
```

### Error 500: Modelo No Encontrado
```json
{
  "predictions": {
    "72h": {
      "value": null,
      "error": "Model not found. Please train model first.",
      "predicted_time": null
    }
  }
}
```

---

## 🎯 Resumen de Endpoints

| Endpoint | Método | Descripción | Respuesta Principal |
|----------|--------|-------------|---------------------|
| `/health` | GET | Estado del servicio | Estado y configuración |
| `/targets` | GET | Listar targets disponibles | Lista de contaminantes |
| `/metrics/{target}/r2` | GET | **NUEVO** R² promedio del target | Métrica de precisión |
| `/predict/{target}/5-horizons` | POST | **NUEVO** Predice 5 horizontes fijos | Predicciones 1,12,24,72,168h |
| `/predict/{target}/risk` | POST | **NUEVO** Clasifica nivel de riesgo | Predicción + clasificación |
| `/predict/{target}` | POST | Predice con horizontes personalizados | Predicciones configurables |
| `/predict` | POST | Endpoint retrocompatible | Default: PM2.5 |
| `/docs` | GET | Documentación interactiva Swagger | UI interactiva |

---

## 🔧 Estructura de Respuestas

### Predicción Estándar (PredictionResponse)

```typescript
{
  target: string,              // Contaminante predicho
  input_time: datetime,        // Momento de inicio de predicción
  predictions: {
    "{H}h": {
      value: number,          // Valor predicho en μg/m³
      predicted_time: datetime, // Momento futuro predicho
      horizon_hours: number   // Horas en el futuro
    }
  },
  unit: string                // Unidad de medida
}
```

### Clasificación de Riesgo (RiskResponse)

```typescript
{
  target: string,             // Contaminante evaluado
  predicted_value: number,    // Valor predicho a 1h
  risk_level: string,         // "Low" | "Medium" | "High"
  unit: string,               // Unidad de medida
  message: string             // Mensaje descriptivo
}
```

### Métricas R²

```typescript
{
  target: string,             // Target consultado
  average_r2: number,         // R² promedio (0-1)
  models_count: number        // Cantidad de modelos
}
```

---

## 📊 Características de la API

✅ **Multi-target**: Soporta PM2.5, PM10, Ozono, NO₂  
✅ **Flexible**: Horizontes configurables por request  
✅ **Métricas de calidad**: R² promedio de modelos  
✅ **Clasificación de riesgo**: Sistema automático de alertas  
✅ **Retrocompatible**: Endpoints legacy funcionan igual  
✅ **Documentación**: Swagger UI en `/docs`  
✅ **Validación**: Pydantic para inputs seguros  
✅ **Manejo de errores**: Respuestas claras y útiles  

**¡La API está lista para producción!** 🚀
