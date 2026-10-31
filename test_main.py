import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "routes" in data
    assert len(data["routes"]) == 10

@pytest.mark.parametrize("agent_idx", range(1, 11))
def test_agent_endpoints(agent_idx):
    payload = {
        "prompt": f"Test task for Agent {agent_idx}",
        "context": {"priority": "high"},
        "parameters": {"temperature": 0.7}
    }
    response = client.post(f"/agent{agent_idx}", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == f"agent{agent_idx}"
    assert data["status"] == "success"
    assert data["input_received"]["prompt"] == payload["prompt"]
    assert f"Agent {agent_idx} executed task successfully" in data["result"]
