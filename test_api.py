"""
Tests for FastAPI Endpoints using TestClient
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "Punjab Agriculture Department" in data["department"]
    assert "version" in data
    assert "uptime" in data
    assert "subsystems" in data
    assert data["subsystems"]["qdrant_vector_db"]["status"] == "operational"
    assert data["subsystems"]["qdrant_vector_db"]["total_points"] >= 7
    assert data["subsystems"]["knowledge_base"]["status"] == "operational"
    assert data["subsystems"]["langgraph_workflow"]["status"] == "operational"
    assert data["subsystems"]["weather_telemetry"]["status"] == "operational"
    assert "system_metrics" in data
    assert len(data["endpoints"]) >= 5

def test_api_districts():
    res = client.get("/api/districts")
    assert res.status_code == 200
    districts = res.json()
    assert len(districts) >= 10
    names = [d["name"] for d in districts]
    assert "Faisalabad" in names
    assert "Multan" in names

def test_api_chat():
    payload = {
        "message": "Meri wheat ki fasal ke pattay peelay ho rahe hain, kya karun?",
        "district": "Faisalabad",
        "crop": "Wheat (Gandum)"
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "top_diagnosis" in data
    assert "distinguishing_test" in data
    assert "differential_diagnosis" in data
    assert "expert_helpline" in data

def test_api_search():
    payload = {
        "query": "wheat yellow rust punjab",
        "crop": "Wheat",
        "max_results": 2
    }
    res = client.post("/api/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)

if __name__ == "__main__":
    test_api_health()
    test_api_districts()
    test_api_chat()
    test_api_search()
    print("[PASS] All FastAPI endpoint tests passed successfully!")
