"""
Internet Searching Agent (WebSearchAgent) for Agricultural Field Assistant
Provides live web search, agricultural news retrieval, and Pakistani government
advisory lookup (e.g. agripunjab.gov.pk, PARC, AARI) with offline resilience.
"""
import re
import urllib.parse
from typing import List, Dict, Any, Optional
import requests

# Try importing ddgs first, then fallback to duckduckgo_search
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False


# Reputable Pakistani agricultural research & governmental domains
PAK_AGRI_DOMAINS = [
    "agripunjab.gov.pk",
    "parc.gov.pk",
    "aari.punjab.gov.pk",
    "uaf.edu.pk",
    "pccc.gov.pk"
]


class WebSearchAgent:
    """
    Autonomous Internet Search Agent specializing in agricultural advisory,
    pest alerts, current weather impact, seed varieties, and government subsidy schemes.
    """
    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }

    def _clean_text(self, text: str) -> str:
        """Strips HTML tags and normalizes whitespace."""
        clean = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", clean).strip()

    def search(
        self,
        query: str,
        crop: Optional[str] = None,
        max_results: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes live web search using DDGS with regional Pakistani agricultural context.
        Falls back to Wikipedia and agricultural scrapers if needed.
        """
        if not query or len(query.strip()) < 2:
            return []

        # Enhance query with agricultural context
        enhanced_query = query.strip()
        crop_clean = (crop or "").replace("(Gandum)", "").replace("(Kapas)", "").replace("(Dhan)", "").strip()
        
        # If crop is known and not in query, augment it
        if crop_clean and crop_clean.lower() not in enhanced_query.lower():
            enhanced_query = f"{crop_clean} {enhanced_query}"

        # If query doesn't specify Pakistan/Punjab, append regional context
        if not any(k in enhanced_query.lower() for k in ["pakistan", "punjab", "lahore", "faisalabad", "multan"]):
            enhanced_query = f"{enhanced_query} Punjab Pakistan agriculture"

        results = []

        # 1. Attempt DDGS text search
        if DDGS_AVAILABLE:
            try:
                with DDGS(timeout=self.timeout) as ddgs:
                    raw_results = list(ddgs.text(
                        enhanced_query,
                        region="pk-en",
                        safesearch="moderate",
                        max_results=max_results
                    ))
                    for item in raw_results:
                        results.append({
                            "title": self._clean_text(item.get("title", "")),
                            "snippet": self._clean_text(item.get("body", "")),
                            "url": item.get("href", ""),
                            "source": "Web Search (DDG Live)"
                        })
            except Exception as e:
                # Log or continue to fallback
                pass

        # 2. If results are still empty, try DDGS news search (especially useful for latest schemes/prices)
        if not results and DDGS_AVAILABLE:
            try:
                with DDGS(timeout=self.timeout) as ddgs:
                    news_results = list(ddgs.news(
                        f"{crop_clean or 'agriculture'} {query} Punjab",
                        region="pk-en",
                        max_results=max_results
                    ))
                    for item in news_results:
                        results.append({
                            "title": self._clean_text(item.get("title", "")),
                            "snippet": self._clean_text(item.get("body", "")),
                            "url": item.get("url", ""),
                            "source": f"News ({item.get('source', 'Live Web')})"
                        })
            except Exception:
                pass

        # 3. Fallback: Wikipedia Scientific OpenSearch (fast & reliable scientific fallback)
        if not results:
            try:
                # Look up scientific or crop term on Wikipedia
                search_term = crop_clean if crop_clean else query.split()[0]
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(search_term)}&limit=2&format=json"
                resp = requests.get(wiki_url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    titles = data[1] if len(data) > 1 else []
                    snippets = data[2] if len(data) > 2 else []
                    urls = data[3] if len(data) > 3 else []
                    for i in range(len(titles)):
                        results.append({
                            "title": titles[i],
                            "snippet": snippets[i] or f"Official encyclopedia entry for {titles[i]} in agriculture.",
                            "url": urls[i],
                            "source": "Wikipedia Agricultural Reference"
                        })
            except Exception:
                pass

        return results[:max_results]

    def search_news(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Specialized news search for current wheat support price, Kisan Card, subsidies, and pest alerts.
        """
        if not DDGS_AVAILABLE:
            return []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                items = list(ddgs.news(f"{query} Punjab Pakistan", region="pk-en", max_results=max_results))
                return [
                    {
                        "title": self._clean_text(item.get("title", "")),
                        "snippet": self._clean_text(item.get("body", "")),
                        "url": item.get("url", ""),
                        "date": item.get("date", ""),
                        "source": item.get("source", "Pakistan Agri News")
                    }
                    for item in items
                ]
        except Exception:
            return []

    def summarize_findings(self, results: List[Dict[str, Any]]) -> str:
        """
        Synthesizes web search results into a concise 2-3 sentence agricultural summary.
        """
        if not results:
            return "No external web advisories found."

        snippets = [f"• {r['title']}: {r['snippet']}" for r in results if r.get("snippet")]
        return "\n".join(snippets[:3])


# Singleton instance
web_search_agent = WebSearchAgent()

