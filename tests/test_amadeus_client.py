from unittest.mock import patch, MagicMock
from services.amadeus_client import AmadeusClient
import pytest

@pytest.fixture
def client():
    return AmadeusClient(client_id="test", client_secret="test")

def test_auth_token_request(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok123", "expires_in": 1799}
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mock_resp):
        client._authenticate()
        assert client._token == "tok123"

def test_search_flights_params(client):
    client._token = "tok123"
    client._token_expiry = 9999999999.0
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"price": {"total": "15000"}}]}
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=mock_resp) as mock_get:
        client.search_flights("BLR", "NRT", "2026-07-01", "INR")
        params = mock_get.call_args[1]["params"]
        assert params["originLocationCode"] == "BLR"
        assert params["currencyCode"] == "INR"
