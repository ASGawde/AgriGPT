import requests
from backend.core.config import settings

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_current_weather(lat: float, lon: float) -> dict:
    if not settings.OPENWEATHER_API_KEY:
        return {"error": "Weather service not configured"}

    params = {
        "lat": lat,
        "lon": lon,
        "appid": settings.OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        res = requests.get(OPENWEATHER_URL, params=params, timeout=5)
        data = res.json()

        if res.status_code != 200:
            return {"error": "Unable to fetch weather"}

        condition_raw = data["weather"][0]["main"].lower()

        condition = (
            "rainy" if "rain" in condition_raw else
            "cloudy" if "cloud" in condition_raw else
            "sunny"
        )

        return {
            "location": data.get("name", "Your Location"),
            "temp": round(data["main"]["temp"]),
            "humidity": data["main"]["humidity"],
            "wind": round(data["wind"]["speed"] * 3.6), 
            "condition": condition,
        }

    except Exception:
        return {"error": "Weather API unavailable"}
