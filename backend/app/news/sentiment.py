"""News sentiment: Bullish / Bearish / Neutral impact from crypto headlines.

Pulls recent headlines from CryptoCompare's public news feed (no key needed)
and scores them with a crypto-tuned lexicon. Best-effort: any failure returns
NEUTRAL so signal generation is never blocked on news availability.
"""
from __future__ import annotations

import time

import httpx

from app.core.logging import get_logger
from app.schemas import NewsSentiment

log = get_logger(__name__)

NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

BULLISH = {
    "surge", "soar", "rally", "bull", "bullish", "gain", "rise", "breakout",
    "adoption", "partnership", "upgrade", "approval", "etf", "inflow", "record",
    "all-time high", "ath", "buy", "accumulate", "support", "recover",
}
BEARISH = {
    "crash", "plunge", "dump", "bear", "bearish", "fall", "drop", "selloff",
    "hack", "exploit", "ban", "lawsuit", "sec", "liquidation", "outflow",
    "downtrend", "fear", "fud", "reject", "collapse", "warning", "scam",
}

# coin -> keywords used to attribute a headline to a symbol
COIN_KEYWORDS = {
    "BTC": {"bitcoin", "btc"},
    "ETH": {"ethereum", "eth", "ether"},
    "SOL": {"solana", "sol"},
    "BNB": {"binance coin", "bnb"},
    "XRP": {"ripple", "xrp"},
}


class NewsAnalyzer:
    def __init__(self, ttl: float = 300.0) -> None:
        self._ttl = ttl
        self._cache: tuple[float, list[dict]] | None = None

    async def _headlines(self) -> list[dict]:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < self._ttl:
            return self._cache[1]
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(NEWS_URL)
                r.raise_for_status()
                items = r.json().get("Data", [])
        except Exception as exc:
            log.debug("news fetch failed: %s", exc)
            items = []
        self._cache = (now, items)
        return items

    async def analyze(self, symbol: str) -> tuple[NewsSentiment, float, list[str]]:
        """Return (sentiment, score[-1..1], sample_headlines) for a symbol."""
        base = symbol.split("/")[0].upper()
        keywords = COIN_KEYWORDS.get(base, {base.lower()})
        items = await self._headlines()

        score, matched = 0.0, []
        for it in items[:60]:
            text = f"{it.get('title', '')} {it.get('body', '')}".lower()
            if not any(k in text for k in keywords):
                continue
            title = it.get("title", "")
            hits = sum(w in text for w in BULLISH) - sum(w in text for w in BEARISH)
            if hits != 0:
                score += hits
                if len(matched) < 5:
                    matched.append(title)

        norm = max(-1.0, min(1.0, score / 6.0))
        if norm > 0.15:
            sentiment = NewsSentiment.BULLISH
        elif norm < -0.15:
            sentiment = NewsSentiment.BEARISH
        else:
            sentiment = NewsSentiment.NEUTRAL
        return sentiment, round(norm, 3), matched


news_analyzer = NewsAnalyzer()
