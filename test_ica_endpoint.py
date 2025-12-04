"""
Test script for the new ICA (Air Quality Index) endpoint
"""

import requests
import json

# API base URL (adjust if needed)
BASE_URL = "http://localhost:8000"

# Sample input data (historical data points)
sample_data = [
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
]

def test_ica_endpoint():
    """Test the /predict/ica endpoint"""
    print("Testing ICA endpoint...")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/ica",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            ica_value = response.json()
            print(f"\n✅ ICA Value: {ica_value}")
            print(f"\nInterpretación:")
            interpretations = {
                1: "Buena",
                2: "Razonablemente buena",
                3: "Regular",
                4: "Desfavorable",
                5: "Muy desfavorable",
                6: "Extremadamente desfavorable"
            }
            print(f"   {interpretations.get(ica_value, 'Desconocido')}")
        else:
            print(f"\n❌ Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar al servidor API.")
        print("   Asegúrese de que el servidor esté corriendo en http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("Test del Endpoint ICA")
    print("=" * 50)
    print()
    test_ica_endpoint()
