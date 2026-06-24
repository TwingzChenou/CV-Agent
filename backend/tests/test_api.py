import pytest
from unittest.mock import patch, AsyncMock

def test_root_endpoint(client):
    """Teste la route racine '/'."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Agent CV" in response.json()["message"]

def test_health_check_endpoint(client):
    """Teste la route de healthcheck '/health'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.api.routers.chat.generate_response", new_callable=AsyncMock)
def test_chat_endpoint_success(mock_generate, client):
    """Teste le bon fonctionnement de la route '/api/chat' avec mock."""
    mock_generate.return_value = "Bonjour, je suis l'assistant de Quentin."
    
    request_payload = {
        "message": "Bonjour, quelle est la disponibilité de Quentin ?",
        "session_id": "test-session-123",
        "user_id": "test-user-456"
    }
    
    response = client.post("/api/chat", json=request_payload)
    
    assert response.status_code == 200
    assert response.json() == {"response": "Bonjour, je suis l'assistant de Quentin."}
    
    # Vérifie que la fonction métier a bien été appelée avec les bons arguments
    mock_generate.assert_called_once_with(
        "Bonjour, quelle est la disponibilité de Quentin ?",
        session_id="test-session-123",
        user_id="test-user-456"
    )

def test_chat_endpoint_validation_error(client):
    """Teste l'erreur de validation (422) si le corps de requête est vide ou incomplet."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422

@patch("app.api.routers.chat.generate_response", new_callable=AsyncMock)
def test_chat_endpoint_rate_limit(mock_generate, client):
    """Teste la limitation de débit (rate limiting) de l'endpoint /api/chat."""
    mock_generate.return_value = "Bonjour."
    request_payload = {
        "message": "Hello",
    }
    
    # On fait 20 requêtes. Au moins une à la fin doit renvoyer 429.
    status_codes = []
    for _ in range(20):
        response = client.post("/api/chat", json=request_payload)
        status_codes.append(response.status_code)
        
    assert 429 in status_codes

