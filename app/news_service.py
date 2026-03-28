"""
Deterministic News & Sentiment Service
---------------------------------------

Code fetches news articles from Exa, LLM only summarizes sentiment.
Single source of truth for all sentiment data in the system.

Used by:
- Workflow step run_news_sentiment()
- /api/price-prediction route
"""

from __future__ import annotations

import json
import logging
import os
import re

from schemas.market import SentimentReport

logger = logging.getLogger(__name__)


def fetch_sentiment(query: str, num_results: int = 5) -> SentimentReport:
    """Fetch news via Exa SDK and summarize sentiment via LLM.

    Code handles: search query, API call, text extraction.
    LLM handles: sentiment scoring from provided text.

    Args:
        query: Search query (e.g. "Bitcoin price prediction crypto")
        num_results: Max articles to fetch from Exa

    Returns:
        SentimentReport — always valid, with fallback on any failure.
    """
    # 1. Fetch articles from Exa (deterministic)
    articles = []
    try:
        from exa_py import Exa

        api_key = os.getenv("EXA_API_KEY", "")
        if api_key:
            exa = Exa(api_key)
            result = exa.search_and_contents(
                query,
                num_results=num_results,
                text={"max_characters": 500},
                type="auto",
            )
            for r in result.results:
                articles.append({
                    "title": r.title or "",
                    "url": r.url or "",
                    "published": getattr(r, "published_date", None),
                    "text": (r.text or "")[:500],
                })
        else:
            logger.warning("EXA_API_KEY not set, skipping news search")
    except Exception as e:
        logger.warning("Exa search failed: %s", e)

    sources_count = len(articles)

    # 2. Build context for LLM
    if articles:
        articles_text = "\n\n".join(
            f"[{i + 1}] {a['title']}\n{a['text']}"
            for i, a in enumerate(articles)
        )
    else:
        articles_text = "No news articles found. Use general crypto market knowledge."

    prompt = (
        f"Analyze sentiment for: {query}\n\n"
        f"Recent news ({sources_count} sources):\n{articles_text}\n\n"
        f"Return a JSON object with exactly these fields:\n"
        f'- "query": "{query}"\n'
        f'- "sentiment_score": float from -1.0 (very bearish) to +1.0 (very bullish)\n'
        f'- "key_narratives": array of 1-5 short strings\n'
        f'- "sources_count": {sources_count}\n'
        f'- "confidence": float 0.0-1.0\n'
        f"\nRespond with ONLY the JSON object, no markdown."
    )

    # 3. LLM summarizes sentiment (OpenAI SDK directly — no Agno wrapper)
    try:
        from openai import OpenAI

        client = OpenAI()  # uses OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw_text = response.choices[0].message.content or ""

        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_text)
        if json_match:
            data = json.loads(json_match.group())
            return SentimentReport(
                query=data.get("query", query),
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                key_narratives=data.get("key_narratives", ["No narratives extracted"]),
                sources_count=data.get("sources_count", sources_count),
                confidence=float(data.get("confidence", 0.3)),
            )
    except Exception as e:
        logger.warning("News sentiment LLM failed: %s", e)

    # 4. Fallback — with real sources_count
    return SentimentReport(
        query=query,
        sentiment_score=0.0,
        key_narratives=[f"Search returned {sources_count} articles but LLM parsing failed"],
        sources_count=sources_count,
        confidence=0.1,
    )
