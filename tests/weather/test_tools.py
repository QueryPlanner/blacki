"""Tests for weather tools."""

from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import httpx
import pytest

from blacki.weather.tools import (
    GEOCODING_API_URL,
    WEATHER_API_URL,
    _geocode_location,
    _get_shared_client,
    get_current_weather,
    get_weather_description,
    get_weather_forecast,
)


@pytest.fixture
def tool_context() -> MagicMock:
    """Provide a mock tool context."""
    context = MagicMock()
    context.user_id = "test_user"
    return context


@pytest.fixture
def mock_httpx_client() -> Any:
    """Provide a mocked httpx.AsyncClient."""
    mock_client = create_autospec(httpx.AsyncClient, spec_set=True, instance=True)
    return mock_client


def test_get_weather_description() -> None:
    """Test WMO code to description mapping."""
    assert get_weather_description(0) == "Clear sky"
    assert get_weather_description(95) == "Thunderstorm"
    assert get_weather_description(999) == "Unknown (999)"


@pytest.mark.asyncio
async def test_get_shared_client() -> None:
    """Test the shared httpx client initialization and reuse."""
    import blacki.weather.tools as wt

    # Reset client
    wt._weather_client = None

    client1 = await _get_shared_client()
    assert isinstance(client1, httpx.AsyncClient)

    client2 = await _get_shared_client()
    assert client1 is client2

    # Cleanup
    await client1.aclose()
    wt._weather_client = None


@pytest.mark.asyncio
async def test_geocode_location_success(mock_httpx_client: Any) -> None:
    """Test successful geocoding."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {
        "results": [
            {
                "name": "London",
                "latitude": 51.50853,
                "longitude": -0.12574,
                "timezone": "Europe/London",
                "country": "United Kingdom",
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_httpx_client.get.return_value = mock_response

    result = await _geocode_location("London", mock_httpx_client)

    assert result == {
        "name": "London",
        "latitude": 51.50853,
        "longitude": -0.12574,
        "timezone": "Europe/London",
        "country": "United Kingdom",
    }
    mock_httpx_client.get.assert_called_once_with(
        GEOCODING_API_URL,
        params={"name": "London", "count": 1, "language": "en", "format": "json"},
    )


@pytest.mark.asyncio
async def test_geocode_location_not_found(mock_httpx_client: Any) -> None:
    """Test geocoding with no results."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {"results": []}
    mock_httpx_client.get.return_value = mock_response

    result = await _geocode_location("UnknownCity123", mock_httpx_client)

    assert result is None


@pytest.mark.asyncio
async def test_geocode_location_fallback_success(mock_httpx_client: Any) -> None:
    """Test geocoding fallback to first segment on comma-separated string."""
    empty_response = MagicMock(spec=httpx.Response)
    empty_response.json.return_value = {"results": []}
    empty_response.raise_for_status.return_value = None

    success_response = MagicMock(spec=httpx.Response)
    success_response.json.return_value = {
        "results": [
            {
                "name": "Viman Nagar",
                "latitude": 18.56848,
                "longitude": 73.91584,
                "timezone": "Asia/Kolkata",
                "country": "India",
            }
        ]
    }
    success_response.raise_for_status.return_value = None

    # First call returns empty, second call returns success
    mock_httpx_client.get.side_effect = [empty_response, success_response]

    result = await _geocode_location("Viman Nagar, Pune", mock_httpx_client)

    assert result == {
        "name": "Viman Nagar",
        "latitude": 18.56848,
        "longitude": 73.91584,
        "timezone": "Asia/Kolkata",
        "country": "India",
    }
    assert mock_httpx_client.get.call_count == 2
    mock_httpx_client.get.assert_any_call(
        GEOCODING_API_URL,
        params={
            "name": "Viman Nagar, Pune",
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    mock_httpx_client.get.assert_any_call(
        GEOCODING_API_URL,
        params={"name": "Viman Nagar", "count": 1, "language": "en", "format": "json"},
    )


@pytest.mark.asyncio
async def test_geocode_location_error(mock_httpx_client: Any) -> None:
    """Test geocoding API error."""
    mock_httpx_client.get.side_effect = httpx.RequestError(
        "Network error", request=MagicMock()
    )  # noqa: E501

    result = await _geocode_location("London", mock_httpx_client)

    assert result is None


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_current_weather_success(
    mock_get_shared: Any, mock_httpx_client: Any, tool_context: Any
) -> None:
    """Test successful current weather fetch."""
    mock_get_shared.return_value = mock_httpx_client

    # Setup geocoding response
    geo_response = MagicMock(spec=httpx.Response)
    geo_response.json.return_value = {
        "results": [
            {
                "name": "London",
                "latitude": 51.50853,
                "longitude": -0.12574,
                "timezone": "Europe/London",
                "country": "United Kingdom",
            }
        ]
    }

    # Setup weather response
    weather_response = MagicMock(spec=httpx.Response)
    weather_response.json.return_value = {
        "current": {
            "temperature_2m": 15.0,
            "relative_humidity_2m": 70,
            "apparent_temperature": 14.5,
            "weather_code": 3,
            "wind_speed_10m": 10.5,
        },
        "current_units": {
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h",
        },
    }

    # Configure client.get to return different responses based on URL
    async def side_effect(url: str, **kwargs: Any) -> Any:
        if url == GEOCODING_API_URL:
            return geo_response
        elif url == WEATHER_API_URL:
            return weather_response
        raise ValueError(f"Unexpected URL: {url}")

    mock_httpx_client.get.side_effect = side_effect

    result = await get_current_weather(tool_context, "London")

    assert result["status"] == "success"
    assert result["location"]["name"] == "London"
    assert result["current_weather"]["temperature"] == "15.0 °C"
    assert result["current_weather"]["condition"] == "Overcast"


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_current_weather_empty_location(
    mock_get_shared: Any, tool_context: Any
) -> None:
    """Test current weather with empty location."""
    result = await get_current_weather(tool_context, "   ")
    assert result["status"] == "error"
    assert "Location cannot be empty" in result["message"]


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_current_weather_geo_fail(
    mock_get_shared: Any, mock_httpx_client: Any, tool_context: Any
) -> None:
    """Test current weather when geocoding fails."""
    mock_get_shared.return_value = mock_httpx_client

    geo_response = MagicMock(spec=httpx.Response)
    geo_response.json.return_value = {"results": []}
    mock_httpx_client.get.return_value = geo_response

    result = await get_current_weather(tool_context, "UnknownCity123")
    assert result["status"] == "error"
    assert "Could not find coordinates" in result["message"]


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_current_weather_api_error(
    mock_get_shared: Any, mock_httpx_client: Any, tool_context: Any
) -> None:
    """Test current weather API error."""
    mock_get_shared.return_value = mock_httpx_client

    # Geocoding succeeds
    geo_response = MagicMock(spec=httpx.Response)
    geo_response.json.return_value = {
        "results": [
            {
                "name": "London",
                "latitude": 51.5,
                "longitude": -0.1,
                "timezone": "GMT",
                "country": "UK",
            }
        ]  # noqa: E501
    }

    async def side_effect(url: str, **kwargs: Any) -> Any:
        if url == GEOCODING_API_URL:
            return geo_response
        elif url == WEATHER_API_URL:
            raise httpx.RequestError("Network error", request=MagicMock())

    mock_httpx_client.get.side_effect = side_effect

    result = await get_current_weather(tool_context, "London")
    assert result["status"] == "error"
    assert "Failed to fetch weather data" in result["message"]


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_weather_forecast_success(
    mock_get_shared: Any, mock_httpx_client: Any, tool_context: Any
) -> None:
    """Test successful forecast fetch."""
    mock_get_shared.return_value = mock_httpx_client

    geo_response = MagicMock(spec=httpx.Response)
    geo_response.json.return_value = {
        "results": [
            {
                "name": "London",
                "latitude": 51.5,
                "longitude": -0.1,
                "timezone": "GMT",
                "country": "UK",
            }
        ]  # noqa: E501
    }

    weather_response = MagicMock(spec=httpx.Response)
    weather_response.json.return_value = {
        "daily": {
            "time": ["2023-10-01", "2023-10-02"],
            "temperature_2m_max": [20.0, 22.0],
            "temperature_2m_min": [10.0, 12.0],
            "precipitation_probability_max": [10, 50],
            "weather_code": [0, 61],
        },
        "daily_units": {
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
            "precipitation_probability_max": "%",
        },
    }

    async def side_effect(url: str, **kwargs: Any) -> Any:
        if url == GEOCODING_API_URL:
            return geo_response
        elif url == WEATHER_API_URL:
            return weather_response

    mock_httpx_client.get.side_effect = side_effect

    result = await get_weather_forecast(tool_context, "London", days=2)
    assert result["status"] == "success"
    assert len(result["forecast"]) == 2
    assert result["forecast"][0]["max_temp"] == "20.0 °C"
    assert result["forecast"][1]["condition"] == "Slight rain"


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_weather_forecast_empty_location(
    mock_get_shared: Any, tool_context: Any
) -> None:
    """Test forecast with empty location."""
    result = await get_weather_forecast(tool_context, "   ")
    assert result["status"] == "error"
    assert "Location cannot be empty" in result["message"]


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_weather_forecast_geo_fail(
    mock_get_shared: Any, mock_httpx_client: Any, tool_context: Any
) -> None:
    """Test forecast when geocoding fails."""
    mock_get_shared.return_value = mock_httpx_client
    geo_response = MagicMock(spec=httpx.Response)
    geo_response.json.return_value = {"results": []}
    mock_httpx_client.get.return_value = geo_response

    result = await get_weather_forecast(tool_context, "UnknownCity")
    assert result["status"] == "error"
    assert "Could not find coordinates" in result["message"]


@pytest.mark.asyncio
@patch("blacki.weather.tools._get_shared_client")
async def test_get_weather_forecast_api_error(
    mock_get_shared: Any, mock_httpx_client: Any, tool_context: Any
) -> None:
    """Test forecast API error."""
    mock_get_shared.return_value = mock_httpx_client

    geo_response = MagicMock(spec=httpx.Response)
    geo_response.json.return_value = {
        "results": [
            {
                "name": "London",
                "latitude": 51.5,
                "longitude": -0.1,
                "timezone": "GMT",
                "country": "UK",
            }
        ]  # noqa: E501
    }

    async def side_effect(url: str, **kwargs: Any) -> Any:
        if url == GEOCODING_API_URL:
            return geo_response
        elif url == WEATHER_API_URL:
            raise httpx.RequestError("Network error", request=MagicMock())

    mock_httpx_client.get.side_effect = side_effect

    result = await get_weather_forecast(tool_context, "London")
    assert result["status"] == "error"
    assert "Failed to fetch weather forecast" in result["message"]
