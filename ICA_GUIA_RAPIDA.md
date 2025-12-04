# 🌡️ Endpoint ICA - Guía Rápida

## 📌 Endpoint
```
POST /predict/ica
```

## 📥 Input
Enviar un JSON con lista de datos históricos (últimas 24h recomendadas):

```json
[
  {
    "time": "2025-12-04T10:00:00",
    "pm2_5": 15.0,
    "pm10": 30.0,
    "nitrogen_dioxide": 40.0,
    "ozone": 70.0,
    "temperature_2m": 23.5,
    "relative_humidity_2m": 52.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 185.0,
    "precipitation": 0.0,
    "surface_pressure": 1014.5
  }
]
```

## 📤 Output
Un número del 1 al 6:

```json
2
```

## 🎨 Interpretación

| Valor | Categoría | Color | Significado |
|-------|-----------|-------|-------------|
| **1** | Buena | 🔵 Azul | Excelente calidad del aire |
| **2** | Razonablemente buena | 🟢 Verde | Calidad aceptable |
| **3** | Regular | 🟡 Amarillo | Grupos sensibles pueden experimentar molestias |
| **4** | Desfavorable | 🔴 Rojo | Efectos en grupos sensibles |
| **5** | Muy desfavorable | 🟤 Granate | Todos pueden experimentar efectos |
| **6** | Extremadamente desfavorable | 🟣 Morado | ⚠️ ALERTA - Efectos graves en la salud |

## 🧪 Prueba Rápida

### Con cURL:
```bash
curl -X POST http://localhost:8000/predict/ica \
  -H "Content-Type: application/json" \
  -d '[{"time":"2025-12-04T10:00:00","pm2_5":15.0,"pm10":30.0,"nitrogen_dioxide":40.0,"ozone":70.0,"temperature_2m":23.5,"relative_humidity_2m":52.0,"wind_speed_10m":5.0,"wind_direction_10m":185.0,"precipitation":0.0,"surface_pressure":1014.5}]'
```

### Con Python:
```python
import requests

data = [{
    "time": "2025-12-04T10:00:00",
    "pm2_5": 15.0,
    "pm10": 30.0,
    "nitrogen_dioxide": 40.0,
    "ozone": 70.0,
    "temperature_2m": 23.5,
    "relative_humidity_2m": 52.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 185.0,
    "precipitation": 0.0,
    "surface_pressure": 1014.5
}]

ica = requests.post("http://localhost:8000/predict/ica", json=data).json()
print(f"ICA: {ica}")
```

### Con el script de ejemplo:
```bash
python ejemplo_ica.py
```

## 📊 Umbrales por Contaminante (μg/m³)

### PM2.5
| ICA | Rango |
|-----|-------|
| 1 | < 10 |
| 2 | 10-20 |
| 3 | 20-25 |
| 4 | 25-50 |
| 5 | 50-75 |
| 6 | ≥ 75 |

### PM10
| ICA | Rango |
|-----|-------|
| 1 | < 20 |
| 2 | 20-40 |
| 3 | 40-50 |
| 4 | 50-100 |
| 5 | 100-150 |
| 6 | ≥ 150 |

### Ozono (O₃)
| ICA | Rango |
|-----|-------|
| 1 | < 50 |
| 2 | 50-100 |
| 3 | 100-130 |
| 4 | 130-240 |
| 5 | 240-380 |
| 6 | ≥ 380 |

### Dióxido de Nitrógeno (NO₂)
| ICA | Rango |
|-----|-------|
| 1 | < 40 |
| 2 | 40-90 |
| 3 | 90-120 |
| 4 | 120-230 |
| 5 | 230-340 |
| 6 | ≥ 340 |

## ⚙️ Cómo Funciona

1. **Predicción individual**: Obtiene predicción a 1h para cada contaminante (PM2.5, PM10, O₃, NO₂)
2. **Cálculo por contaminante**: Calcula ICA individual según umbrales de Orden TEC/351/2019
3. **ICA final**: Retorna el **peor valor** (máximo) entre todos los contaminantes

## 📚 Base Legal
Basado en la **Orden TEC/351/2019** del Ministerio para la Transición Ecológica de España.

## ✅ Iniciar Servidor
```bash
cd c:\Users\mati9\OneDrive\Desktop\Uni\6to Semestre\ModeloPredictivo
python src/api.py
```

## 🔍 Ver Documentación Interactiva
Una vez iniciado el servidor, visita:
```
http://localhost:8000/docs
```
