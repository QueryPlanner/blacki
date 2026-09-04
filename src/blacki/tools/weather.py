"""Weather tools using Open-Meteo."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

_weather_client_lock = asyncio.Lock()
_weather_client: httpx.AsyncClient | None = None

WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather_description(code: int) -> str:
    """Convert WMO code to human-readable description."""
    return WMO_CODES.get(code, f"Unknown ({code})")


async def cleanup_weather_client() -> None:
    """Close the shared HTTP client."""
    global _weather_client
    async with _weather_client_lock:
        if _weather_client is not None:
            await _weather_client.aclose()
            _weather_client = None


async def _get_shared_client() -> httpx.AsyncClient:
    global _weather_client
    async with _weather_client_lock:
        if _weather_client is None:
            _weather_client = httpx.AsyncClient(timeout=10.0)
        return _weather_client


async def _geocode_location(
    location: str, client: httpx.AsyncClient
) -> dict[str, Any] | None:
    """Resolve location to coordinates, falling back to first segment if needed."""

    async def fetch(query: str) -> dict[str, Any] | None:
        params: dict[str, Any] = {
            "name": query.strip(),
            "count": 1,
            "language": "en",
            "format": "json",
        }
        try:
            response = await client.get(GEOCODING_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results")
            if not results:
                return None

            result = results[0]
            return {
                "name": result.get("name"),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "timezone": result.get("timezone", "auto"),
                "country": result.get("country", ""),
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.exception("Geocoding API HTTP error")
            raise
        except httpx.RequestError:
            logger.exception("Geocoding API network error")
            raise

    # First try exact location string
    result = await fetch(location)

    # If no result and location has commas, try the first segment
    if not result and "," in location:
        first_segment = location.split(",")[0]
        result = await fetch(first_segment)

    return result


async def get_current_weather(
    tool_context: ToolContext,
    location: str,
) -> dict[str, Any]:
    """Get the current weather conditions for a specific location.

    Args:
        tool_context: ADK tool context.
        location: The name of the city or location (e.g., "London", "New York, NY").

    Returns:
        Dictionary with status, location, and current weather data.
    """
    _ = tool_context

    if not location.strip():
        return {"status": "error", "message": "Location cannot be empty."}

    try:
        client = await _get_shared_client()
        geo_data = await _geocode_location(location, client)

        if not geo_data:
            return {
                "status": "error",
                "message": f"Could not find coordinates for location: {location}",
                "location": location,
            }

        params: dict[str, Any] = {
            "latitude": geo_data["latitude"],
            "longitude": geo_data["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,weather_code,wind_speed_10m"
            ),
            "timezone": geo_data["timezone"],
        }

        response = await client.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        current_units = data.get("current_units", {})

        weather_code = current.get("weather_code")
        description = (
            get_weather_description(weather_code)
            if weather_code is not None
            else "Unknown"
        )

        return {
            "status": "success",
            "location": {
                "name": geo_data["name"],
                "country": geo_data["country"],
                "latitude": geo_data["latitude"],
                "longitude": geo_data["longitude"],
            },
            "current_weather": {
                "temperature": (
                    f"{current.get('temperature_2m')} "
                    f"{current_units.get('temperature_2m', '°C')}"
                ),
                "feels_like": (
                    f"{current.get('apparent_temperature')} "
                    f"{current_units.get('apparent_temperature', '°C')}"
                ),
                "humidity": (
                    f"{current.get('relative_humidity_2m')} "
                    f"{current_units.get('relative_humidity_2m', '%')}"
                ),
                "wind_speed": (
                    f"{current.get('wind_speed_10m')} "
                    f"{current_units.get('wind_speed_10m', 'km/h')}"
                ),
                "condition": description,
            },
        }

    except (httpx.RequestError, httpx.HTTPStatusError):
        logger.exception("Weather API error")
        return {
            "status": "error",
            "message": "Failed to fetch weather data due to a network or API error.",
            "location": location,
        }


async def get_weather_forecast(
    tool_context: ToolContext,
    location: str,
    days: int = 3,
) -> dict[str, Any]:
    """Get a multi-day weather forecast for a specific location.

    Args:
        tool_context: ADK tool context.
        location: The name of the city or location.
        days: Number of days for the forecast (1 to 14, default is 3).

    Returns:
        Dictionary with status, location, and daily forecast data.
    """
    _ = tool_context

    if not location.strip():
        return {"status": "error", "message": "Location cannot be empty."}

    days = max(1, min(14, days))

    try:
        client = await _get_shared_client()
        geo_data = await _geocode_location(location, client)

        if not geo_data:
            return {
                "status": "error",
                "message": f"Could not find coordinates for location: {location}",
                "location": location,
            }

        params: dict[str, Any] = {
            "latitude": geo_data["latitude"],
            "longitude": geo_data["longitude"],
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "timezone": geo_data["timezone"],
            "forecast_days": days,
        }

        response = await client.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        daily_units = data.get("daily_units", {})

        forecast = []
        times = daily.get("time", [])
        weather_codes = daily.get("weather_code", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip_probs = daily.get("precipitation_probability_max", [])

        for i, t in enumerate(times):
            code = weather_codes[i] if i < len(weather_codes) else None
            max_temp = max_temps[i] if i < len(max_temps) else "N/A"
            min_temp = min_temps[i] if i < len(min_temps) else "N/A"
            precip_prob = precip_probs[i] if i < len(precip_probs) else "N/A"

            unit_max = daily_units.get("temperature_2m_max", "°C")
            unit_min = daily_units.get("temperature_2m_min", "°C")
            unit_precip = daily_units.get("precipitation_probability_max", "%")
            forecast.append(
                {
                    "date": t,
                    "max_temp": f"{max_temp} {unit_max}",
                    "min_temp": f"{min_temp} {unit_min}",
                    "precipitation_probability": f"{precip_prob}{unit_precip}",
                    "condition": (
                        get_weather_description(code) if code is not None else "Unknown"
                    ),
                }
            )

        return {
            "status": "success",
            "location": {
                "name": geo_data["name"],
                "country": geo_data["country"],
            },
            "forecast": forecast,
        }

    except (httpx.RequestError, httpx.HTTPStatusError):
        logger.exception("Weather API error")
        return {
            "status": "error",
            "message": (
                "Failed to fetch weather forecast due to a network or API error."
            ),
            "location": location,
        }
