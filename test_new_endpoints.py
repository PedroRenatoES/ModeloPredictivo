import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_r2():
    print("\nTesting R2 Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics/pm2_5/r2")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_5_horizons():
    print("\nTesting 5-Horizons Endpoint...")
    data = [
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
    try:
        response = requests.post(f"{BASE_URL}/predict/pm2_5/5-horizons", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_risk():
    print("\nTesting Risk Endpoint...")
    data = [
        {
            "time": "2025-07-01T12:00:00",
            "pm2_5": 15.5, # Low value
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
        response = requests.post(f"{BASE_URL}/predict/pm2_5/risk", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait a bit for server to start
    time.sleep(2)
    
    with open("test_results.txt", "w") as f:
        import sys
        sys.stdout = f
        test_r2()
        test_5_horizons()
        test_risk()
