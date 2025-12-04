"""
Ejemplo de uso del endpoint ICA - Índice de Calidad del Aire

Este script demuestra cómo usar el nuevo endpoint /predict/ica
"""

import requests
import json
from datetime import datetime

# URL base de la API
#BASE_URL = "http://localhost:8000"
BASE_URL = "https://modelopredictivo.onrender.com/:8000"

# Datos de ejemplo (condiciones actuales de calidad del aire)
current_conditions = [
    {
        "time": "2025-12-04T09:00:00",
        "pm2_5": 12.5,
        "pm10": 28.0,
        "nitrogen_dioxide": 35.0,
        "ozone": 65.0,
        "temperature_2m": 22.0,
        "relative_humidity_2m": 55.0,
        "wind_speed_10m": 4.5,
        "wind_direction_10m": 180.0,
        "precipitation": 0.0,
        "surface_pressure": 1015.0
    },
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

def get_ica_description(ica_value):
    """Retorna la descripción del ICA"""
    descriptions = {
        1: {
            "nivel": "Buena",
            "color": "Azul",
            "recomendacion": "La calidad del aire es buena. Disfrute de actividades al aire libre."
        },
        2: {
            "nivel": "Razonablemente buena",
            "color": "Verde",
            "recomendacion": "La calidad del aire es aceptable. Puede realizar actividades al aire libre."
        },
        3: {
            "nivel": "Regular",
            "color": "Amarillo",
            "recomendacion": "Grupos sensibles pueden experimentar molestias. Limite actividades prolongadas al aire libre."
        },
        4: {
            "nivel": "Desfavorable",
            "color": "Rojo",
            "recomendacion": "Grupos sensibles deben reducir el ejercicio al aire libre. El público general puede experimentar molestias."
        },
        5: {
            "nivel": "Muy desfavorable",
            "color": "Granate",
            "recomendacion": "Todos pueden experimentar efectos en la salud. Evite actividades al aire libre."
        },
        6: {
            "nivel": "Extremadamente desfavorable",
            "color": "Morado",
            "recomendacion": "⚠️ ALERTA DE SALUD: Todos pueden experimentar efectos graves. Permanezca en interiores."
        }
    }
    return descriptions.get(ica_value, {"nivel": "Desconocido", "color": "Gris", "recomendacion": "No disponible"})

def calculate_ica():
    """Calcula el ICA usando el endpoint"""
    print("=" * 70)
    print("CÁLCULO DEL ÍNDICE DE CALIDAD DEL AIRE (ICA)")
    print("=" * 70)
    print()
    
    try:
        # Realizar petición al endpoint
        print("📡 Consultando API...")
        response = requests.post(
            f"{BASE_URL}/predict/ica",
            json=current_conditions,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            ica_value = response.json()
            info = get_ica_description(ica_value)
            
            print("✅ ICA calculado exitosamente\n")
            print("━" * 70)
            print(f"📊 VALOR ICA: {ica_value}")
            print(f"🏷️  CATEGORÍA: {info['nivel']}")
            print(f"🎨 COLOR: {info['color']}")
            print("━" * 70)
            print(f"\n💡 Recomendación:")
            print(f"   {info['recomendacion']}")
            print()
            
            # Mostrar detalles adicionales
            print("📋 Detalles de las condiciones actuales:")
            last_reading = current_conditions[-1]
            print(f"   • PM2.5: {last_reading['pm2_5']} μg/m³")
            print(f"   • PM10: {last_reading['pm10']} μg/m³")
            print(f"   • NO₂: {last_reading['nitrogen_dioxide']} μg/m³")
            print(f"   • O₃: {last_reading['ozone']} μg/m³")
            print(f"   • Temperatura: {last_reading['temperature_2m']}°C")
            print(f"   • Humedad: {last_reading['relative_humidity_2m']}%")
            print()
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar al servidor API")
        print("\n⚠️  Asegúrese de que el servidor esté corriendo:")
        print("   python src/api.py")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

def show_ica_scale():
    """Muestra la escala del ICA"""
    print("\n" + "=" * 70)
    print("ESCALA DEL ÍNDICE DE CALIDAD DEL AIRE (ICA)")
    print("=" * 70)
    print()
    for i in range(1, 7):
        info = get_ica_description(i)
        print(f"{i} - {info['nivel']:30s} ({info['color']})")
    print()

if __name__ == "__main__":
    show_ica_scale()
    calculate_ica()
    print("=" * 70)
