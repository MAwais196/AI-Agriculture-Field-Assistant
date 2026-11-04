"""
Tests for Internet Searching Agent (WebSearchAgent)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search_agent import web_search_agent

def test_web_search_agent_search():
    results = web_search_agent.search("wheat yellow rust", crop="Wheat", max_results=3)
    assert isinstance(results, list)
    # Even if offline, search should return list without crashing
    if len(results) > 0:
        first = results[0]
        assert "title" in first
        assert "url" in first or "snippet" in first
        assert "source" in first

def test_web_search_agent_summary():
    mock_results = [
        {"title": "Punjab Wheat Advisory 2026", "snippet": "Yellow rust surveillance active in Faisalabad and Sahiwal.", "url": "https://agripunjab.gov.pk"},
        {"title": "Fungicide Guidelines", "snippet": "Spray Nativo or Tilt on initial rust appearance.", "url": "https://aari.punjab.gov.pk"}
    ]
    summary = web_search_agent.summarize_findings(mock_results)
    assert "Punjab Wheat Advisory" in summary
    assert "Fungicide Guidelines" in summary

def test_web_search_news():
    news = web_search_agent.search_news("wheat punjab", max_results=2)
    assert isinstance(news, list)

if __name__ == "__main__":
    test_web_search_agent_search()
    test_web_search_agent_summary()
    test_web_search_news()
    print("[PASS] All Internet Search Agent tests passed successfully!")

