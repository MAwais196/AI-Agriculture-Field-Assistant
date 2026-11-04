"""
Tests for Weather Service and Epidemiological Heuristics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.weather_service import weather_service

def test_weather_retrieval_faisalabad():
    res = weather_service.get_weather_for_district("faisalabad")
    assert res["district"] == "Faisalabad"
    assert "temperature" in res
    assert "humidity" in res
    assert "risk_assessment" in res

def test_weather_retrieval_multan():
    res = weather_service.get_weather_for_district("multan")
    assert res["district"] == "Multan"
    assert res["zone"] == "Cotton-Wheat Zone"

if __name__ == "__main__":
    test_weather_retrieval_faisalabad()
    test_weather_retrieval_multan()
    print("[PASS] All Weather tests passed successfully!")
