"""
Location: agents/web_search.py

Purpose:
    Tavily web search used as the RAG agent's fallback when the knowledge
    base does not cover a question. Titles, URLs and dates come only from
    the Tavily API response; nothing here is generated or invented.
"""

from typing import Any, Dict, List
from urllib.parse import urlparse
import logging
import os

import httpx

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

# Regulator, government and DISCO domains, ranked ahead of other results.
OFFICIAL_DOMAINS = (
    "nepra.org.pk", "ke.com.pk", "iesco.com.pk", "fesco.com.pk",
    "gepco.com.pk", "mepco.com.pk", "sepco.com.pk", "qesco.com.pk",
)

# Forums and social media are not evidence, however relevant.
EXCLUDED_DOMAINS = (
    "reddit.com", "facebook.com", "quora.com", "x.com", "twitter.com",
    "youtube.com", "tiktok.com", "instagram.com", "pinterest.com",
)


def is_official(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host.endswith(".gov.pk") or any(
        host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS
    )


def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web via Tavily. Returns [] when no key is configured or the
    request fails, so callers can report that web research did not happen.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; web search skipped.")
        return []

    min_score = float(os.getenv("WEB_MIN_SCORE", "0.3"))

    try:
        resp = httpx.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": f"{query} Pakistan electricity",
                "max_results": max_results,
                "search_depth": "basic",
                "topic": "general",
                "exclude_domains": list(EXCLUDED_DOMAINS),
            },
            timeout=8,
        )
        resp.raise_for_status()
        raw = resp.json().get("results") or []
    except Exception:
        logger.exception("Tavily search failed.")
        return []

    results = []
    for r in raw:
        url = str(r.get("url") or "")
        if not url.startswith(("http://", "https://")):
            continue
        if float(r.get("score") or 0) < min_score:
            continue
        results.append({
            "title": str(r.get("title") or urlparse(url).hostname),
            "url": url,
            "content": str(r.get("content") or "")[:700],
            "published_date": r.get("published_date") or "",
            "official": is_official(url),
        })

    # Stable sort keeps Tavily's relevance order within each group.
    results.sort(key=lambda r: not r["official"])
    return results
