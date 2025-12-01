# API Multi-Target - Guía de Uso

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

### 3. Predecir para un Target Específico
```http
POST /predict/{target}?horizons=1,12,24
```

**Parámetros:**
- `target` (path): pm2_5, pm10, ozone, nitrogen_dioxide
- `horizons` (query, opcional): horizontes separados por comas (ej: "1,12,24")

**Body (JSON):**
```json
{
  "time": "2025-11-30T18:00:00",
  "pm2_5": 15.5,
  "pm10": 25.0,
  "nitrogen_dioxide": 20.0,
  "ozone": 45.0,
  "temperature_2m": 25.0,
  "relative_humidity_2m": 60,
  "wind_speed_10m": 5.5,
  "wind_direction_10m": 180,
  "precipitation": 0.0,
  "surface_pressure": 1013.0
}
```

**Respuesta:**
```json
{
  "target": "ozone",
  "input_time": "2025-11-30T18:00:00",
  "predictions": {
    "1h": {
      "value": 47.82,
      "predicted_time": "2025-11-30T19:00:00",
      "horizon_hours": 1
    },
    "12h": {
      "value": 38.15,
      "predicted_time": "2025-12-01T06:00:00",
      "horizon_hours": 12
    },
    "24h": {
      "value": 52.34,
      "predicted_time": "2025-12-01T18:00:00",
      "horizon_hours": 24
    }
  },
  "unit": "μg/m³"
}
```

---

### 4. Endpoint Compatible con Versión Anterior
```http
POST /predict?target=pm2_5&horizons=1,24
```

**Body:** Mismo formato que arriba

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Predecir Ozono (curl)

```bash
curl -X POST "http://localhost:8000/predict/ozone?horizons=1,12,24" \
  -H "Content-Type: application/json" \
  -d '{
    "time": "2025-11-30T18:00:00",
    "pm2_5": 15.5,
    "pm10": 25.0,
    "nitrogen_dioxide": 20.0,
    "ozone": 45.0,
    "temperature_2m": 25.0,
    "relative_humidity_2m": 60,
    "wind_speed_10m": 5.5,
    "wind_direction_10m": 180,
    "precipitation": 0.0,
    "surface_pressure": 1013.0
  }'
```

### Ejemplo 2: Predecir PM2.5 (Python - requests)

```python
import requests
from datetime import datetime

url = "http://localhost:8000/predict/pm2_5"
params = {"horizons": "1,24,72"}

data = {
    "time": datetime.now().isoformat(),
    "pm2_5": 15.5,
    "pm10": 25.0,
    "nitrogen_dioxide": 20.0,
    "ozone": 45.0,
    "temperature_2m": 25.0,
    "relative_humidity_2m": 60,
    "wind_speed_10m": 5.5,
    "wind_direction_10m": 180,
    "precipitation": 0.0,
    "surface_pressure": 1013.0
}

response = requests.post(url, params=params, json=data)
predictions = response.json()

print(f"Predicciones para {predictions['target']}:")
for horizon, pred in predictions['predictions'].items():
    print(f"  {horizon}: {pred['value']} {predictions['unit']}")
```

### Ejemplo 3: Predecir NO₂ (JavaScript - fetch)

```javascript
const predictNO2 = async () => {
  const response = await fetch('http://localhost:8000/predict/nitrogen_dioxide?horizons=1,12', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      time: new Date().toISOString(),
      pm2_5: 15.5,
      pm10: 25.0,
      nitrogen_dioxide: 20.0,
      ozone: 45.0,
      temperature_2m: 25.0,
      relative_humidity_2m: 60,
      wind_speed_10m: 5.5,
      wind_direction_10m: 180,
      precipitation: 0.0,
      surface_pressure: 1013.0
    })
  });

  const data = await response.json();
  console.log('NO₂ Predictions:', data);
};
```

---

## 🔄 Casos de Uso Comunes

### Caso 1: Dashboard con Múltiples Contaminantes

```python
import requests

targets = ["pm2_5", "ozone", "nitrogen_dioxide"]
base_url = "http://localhost:8000/predict"

current_conditions = {
    "time": "2025-11-30T18:00:00",
    "pm2_5": 15.5,
    "pm10": 25.0,
    "nitrogen_dioxide": 20.0,
    "ozone": 45.0,
    "temperature_2m": 25.0,
    "relative_humidity_2m": 60,
    "wind_speed_10m": 5.5,
    "wind_direction_10m": 180,
    "precipitation": 0.0,
    "surface_pressure": 1013.0
}

all_predictions = {}
for target in targets:
    response = requests.post(
        f"{base_url}/{target}",
        params={"horizons": "1,24"},
        json=current_conditions
    )
    all_predictions[target] = response.json()

# Mostrar resultados
for target, result in all_predictions.items():
    print(f"\n{target.upper()}")
    for horizon, pred in result['predictions'].items():
        print(f"  {horizon}: {pred['value']} {result['unit']}")
```

### Caso 2: Alertas Automáticas

```python
import requests

def check_ozone_alert(current_data):
    """Verifica si se espera ozono alto en las próximas 24h"""
    response = requests.post(
        "http://localhost:8000/predict/ozone",
        params={"horizons": "1,12,24"},
        json=current_data
    )
    
    predictions = response.json()
    threshold = 100  # μg/m³
    
    alerts = []
    for horizon, pred in predictions['predictions'].items():
        if pred['value'] > threshold:
            alerts.append({
                'horizon': horizon,
                'value': pred['value'],
                'time': pred['predicted_time']
            })
    
    return alerts
```

---

## ⚠️ Manejo de Errores

### Error 400: Target Inválido
```json
{
  "detail": "Invalid target. Must be one of: ['pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide']"
}
```

### Error 400: Horizons Inválidos
```json
{
  "detail": "Horizons must be comma-separated integers (e.g., '1,12,24')"
}
```

### Error 500: Modelo No Encontrado
```json
{
  "target": "ozone",
  "predictions": {
    "72h": {
      "value": null,
      "error": "Model not found. Please train model first.",
      "predicted_time": null
    }
  }
}
```

**Solución:** Entrenar el modelo primero
```bash
python main.py --target ozone --horizons 72
```

---

## 🔧 Configuración y Deployment

### Development
```bash
python src/api.py
```

### Production con Gunicorn
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app --bind 0.0.0.0:8000
```

### Docker (Ejemplo)
```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 Características de la API

✅ **Multi-target**: Soporta PM2.5, PM10, Ozono, NO₂  
✅ **Flexible**: Horizontes configurables por request  
✅ **Retrocompatible**: Endpoint `/predict` funciona igual  
✅ **Documentación**: Swagger UI en `/docs`  
✅ **Validación**: Pydantic para validar inputs  
✅ **Manejo de errores**: Respuestas claras y útiles  

---

## 🎯 Resumen de Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servicio |
| `/targets` | GET | Listar targets disponibles |
| `/predict/{target}` | POST | Predecir target específico |
| `/predict` | POST | Endpoint retrocompatible (default: PM2.5) |
| `/docs` | GET | Documentación interactiva |

**¡La API está lista para producción!** 🚀
