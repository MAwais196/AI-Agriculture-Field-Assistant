"""
Weather Service using Open-Meteo API
Provides real-time temperature, humidity, wind, and rain data for Punjab districts.
These meteorological conditions are critical epidemiological factors for crop diseases:
- Yellow/Stripe Rust (Wheat): thrives in cool (10-20°C) and high humidity / morning dew (>70%).
- Aphids (Sust Tela): thrive in dry, moderately warm conditions.
- Leaf Blight: thrives in warm, humid conditions.
- Heat stress: temperatures > 32°C during wheat grain filling causes forced maturity and yellowing.
"""
import requests
from typing import Dict, Any, Optional
from app.config import PUNJAB_DISTRICTS

class WeatherService:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    def get_weather_for_district(self, district_key: str) -> Dict[str, Any]:
        """
        Fetches current weather and 3-day forecast for a given Punjab district.
        """
        dist_info = PUNJAB_DISTRICTS.get(district_key.lower().replace(" ", "_"))
        if not dist_info:
            # Fallback to central Punjab (Faisalabad)
            dist_info = PUNJAB_DISTRICTS["faisalabad"]
            district_key = "faisalabad"

        params = {
            "latitude": dist_info["lat"],
            "longitude": dist_info["lon"],
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Karachi"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=6)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                daily = data.get("daily", {})
                
                temp = current.get("temperature_2m")
                humidity = current.get("relative_humidity_2m")
                precip = current.get("precipitation", 0)
                wind = current.get("wind_speed_10m", 0)
                
                # Disease risk epidemiological heuristics
                risk_factors = []
                if humidity is not None and temp is not None:
                    if 10 <= temp <= 22 and humidity >= 65:
                        risk_factors.append("HIGH_YELLOW_RUST_RISK (High humidity + 10-22°C favors Puccinia striiformis)")
                    if temp > 28 and humidity < 50:
                        risk_factors.append("APHID_INFESTATION_RISK (Warm and dry weather favors wheat aphids / tela)")
                    if precip > 5:
                        risk_factors.append("WATERLOGGING_ROOT_ROT_RISK (Recent heavy showers can induce leaf yellowing due to suffocation)")
                    if temp > 33:
                        risk_factors.append("HEAT_STRESS_CHLOROSIS (High ambient temperature causing leaf scorch/drying)")

                return {
                    "district": dist_info["name"],
                    "zone": dist_info["zone"],
                    "temperature": temp,
                    "humidity": humidity,
                    "precipitation_mm": precip,
                    "wind_speed_kmh": wind,
                    "risk_assessment": risk_factors,
                    "summary": f"{dist_info['name']}: {temp}°C, Humidity {humidity}%, Rain {precip}mm, Wind {wind} km/h"
                }
        except Exception as e:
            # Fallback mock weather if offline
            return {
                "district": dist_info["name"],
                "zone": dist_info["zone"],
                "temperature": 18.5,
                "humidity": 72.0,
                "precipitation_mm": 0.0,
                "wind_speed_kmh": 8.0,
                "risk_assessment": ["HIGH_YELLOW_RUST_RISK (High humidity + 10-22°C favors Puccinia striiformis)"],
                "summary": f"{dist_info['name']} (Simulated): 18.5°C, Humidity 72%, Wind 8 km/h"
            }

weather_service = WeatherService()

