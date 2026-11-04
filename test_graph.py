"""
Tests for LangGraph State Machine Execution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.decision_graph import field_agent_graph

def test_graph_yellow_rust_flow():
    initial_state = {
        "session_id": "test_case_rust",
        "user_message": "Meri wheat ki fasal ke pattay peelay ho rahe hain aur hath laganay par haldi jaisa powder lagta hai",
        "crop": "Wheat (Gandum)",
        "district": "Faisalabad",
        "crop_stage": "Heading (Sitta nikalna)",
        "audio_path": None,
        "image_path": None,
        "symptoms": []
    }

    result = field_agent_graph.invoke(initial_state)

    assert result["top_diagnosis"] is not None
    assert "Yellow Rust" in result["top_diagnosis"] or "Rust" in result["top_diagnosis"]
    assert result["confidence_score"] >= 0.60
    assert "FINGER RUB TEST" in result["distinguishing_test"]
    assert len(result["official_sources"]) > 0
    assert "0800-15000" in result["expert_helpline"]

def test_graph_nitrogen_flow():
    initial_state = {
        "session_id": "test_case_nitrogen",
        "user_message": "Wheat ke neechay walay puraney pattay peelay ho rahe hain, koi powder nahi hai",
        "crop": "Wheat (Gandum)",
        "district": "Sahiwal",
        "crop_stage": "Tillering",
        "audio_path": None,
        "image_path": None,
        "symptoms": []
    }

    result = field_agent_graph.invoke(initial_state)

    assert result["top_diagnosis"] is not None
    assert "Nitrogen" in result["top_diagnosis"] or "kami" in result["top_diagnosis"].lower()
    assert len(result["recommended_actions"]) > 0

if __name__ == "__main__":
    test_graph_yellow_rust_flow()
    test_graph_nitrogen_flow()
    print("[PASS] All LangGraph decision tests passed successfully!")
