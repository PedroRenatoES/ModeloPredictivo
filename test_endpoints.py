# -*- coding: utf-8 -*-
import sys
import io
# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

# Sample data for testing
SAMPLE_DATA = [
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

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_health():
    print_section("TEST 1: Health Check Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"[X] Error: {e}")
        return False

def test_targets():
    print_section("TEST 2: Available Targets Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/targets")
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"[X] Error: {e}")
        return False

def test_r2_metrics():
    print_section("TEST 3: Average R2 Metrics Endpoint")
    targets = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"]
    
    for target in targets:
        print(f"\n[*] Testing R2 for {target}:")
        try:
            response = requests.get(f"{BASE_URL}/metrics/{target}/r2")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] R2 Promedio: {data['r2_promedio']}")
                print(f"   Cantidad de Modelos: {data['cantidad_modelos']}")
            else:
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"[X] Error: {e}")
    return True

def test_5_horizons():
    print_section("TEST 4: 5-Horizons Prediction Endpoint")
    target = "pm2_5"
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/{target}/5-horizons",
            json=SAMPLE_DATA
        )
        print(f"Target: {target}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Predicciones para {target}:")
            print(f"   Tiempo de Entrada: {data['tiempo_entrada']}")
            print(f"   Unidad: {data['unidad']}")
            print("\n   Horizontes:")
            for horizon, pred in data['predicciones'].items():
                print(f"     {horizon}: {pred['valor']} en {pred['tiempo_predicho']}")
            return True
        else:
            print(f"[X] Response: {response.text}")
            return False
    except Exception as e:
        print(f"[X] Error: {e}")
        return False

def test_risk_classification():
    print_section("TEST 5: Risk Classification Endpoint")
    
    # Test with different pollutant levels
    test_cases = [
        ("Riesgo Bajo", 15.5),   # Low value
        ("Riesgo Medio", 45.0),  # Medium value
        ("Riesgo Alto", 60.0)    # High value
    ]
    
    for case_name, pm25_value in test_cases:
        print(f"\n[*] Testeando {case_name} (PM2.5 = {pm25_value}):")
        
        test_data = [
            {
                "time": "2025-07-01T12:00:00",
                "pm2_5": pm25_value,
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
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict/pm2_5/risk",
                json=test_data
            )
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   [OK] Valor Predicho: {data['valor_predicho']} {data['unidad']}")
                print(f"   [OK] Nivel de Riesgo: {data['nivel_riesgo']}")
                print(f"   [OK] Mensaje: {data['mensaje']}")
            else:
                print(f"   [X] Response: {response.text}")
        except Exception as e:
            print(f"   [X] Error: {e}")
    
    return True

def test_standard_prediction():
    print_section("TEST 6: Standard Prediction Endpoint")
    target = "pm2_5"
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/{target}",
            json=SAMPLE_DATA,
            params={"horizons": "1,24,72"}
        )
        print(f"Target: {target}")
        print(f"Custom Horizons: 1,24,72")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Predicciones:")
            for horizon, pred in data['predicciones'].items():
                print(f"   {horizon}: {pred['valor']} en {pred['tiempo_predicho']}")
            return True
        else:
            print(f"[X] Response: {response.text}")
            return False
    except Exception as e:
        print(f"[X] Error: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("  AIR QUALITY API - ENDPOINTS TEST SUITE")
    print("="*80)
    print(f"\nBase URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(("Health Check", test_health()))
    results.append(("Available Targets", test_targets()))
    results.append(("R2 Metrics", test_r2_metrics()))
    results.append(("5-Horizons Prediction", test_5_horizons()))
    results.append(("Risk Classification", test_risk_classification()))
    results.append(("Standard Prediction", test_standard_prediction()))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
