"""Market-data service: unified OHLCV + open-interest access via ccxt.

One async client per exchange id, cached. Supports the required venues
(Binance/Bybit/OKX/Bitget/BingX) through ccxt's unified API, plus a short
in-memory TTL cache so a burst of requests doesn't hammer the exchange.
"""
from __future__ import annotations

import time

import ccxt.async_support as ccxt
import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# ccxt id aliases for the exchanges named in the requirements
EXCHANGE_IDS = {
    "binance": "binanceusdm",
    "binanceusdm": "binanceusdm",
    "bybit": "bybit",
    "okx": "okx",
    "bitget": "bitget",
    "bingx": "bingx",
}


class MarketData:
    def __init__(self) -> None:
        self._clients: dict[str, ccxt.Exchange] = {}
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 5.0  # seconds

    def _client(self, exchange: str) -> ccxt.Exchange:
        ex_id = EXCHANGE_IDS.get(exchange, exchange)
        if ex_id not in self._clients:
            klass = getattr(ccxt, ex_id)
            self._clients[ex_id] = klass({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        return self._clients[ex_id]

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, *, exchange: str | None = None,
        limit: int | None = None, with_open_interest: bool = True,
    ) -> pd.DataFrame:
        exchange = exchange or settings.default_exchange
        limit = limit or settings.ohlcv_limit
        key = (exchange, symbol, timeframe, limit)
        now = time.monotonic()
        if key in self._cache and now - self._cache[key][0] < self._cache_ttl:
            return self._cache[key][1].copy()

        client = self._client(exchange)
        raw = await client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume"]})

        if with_open_interest:
            df["open_interest"] = await self._fetch_oi_series(client, symbol, timeframe, len(df))

        self._cache[key] = (now, df)
        return df.copy()

    async def _fetch_oi_series(self, client, symbol, timeframe, length) -> pd.Series:
        """Best-effort open-interest history; NaN when the venue doesn't expose it."""
        try:
            if client.has.get("fetchOpenInterestHistory"):
                hist = await client.fetch_open_interest_history(symbol, timeframe, limit=length)
                vals = [h.get("openInterestAmount") or h.get("openInterestValue") for h in hist]
                s = pd.Series(vals, dtype="float64")
                return s.reindex(range(length)).ffill().bfill()
        except Exception as exc:
            log.debug("OI history unavailable for %s: %s", symbol, exc)
        return pd.Series([float("nan")] * length)

    async def fetch_ticker(self, symbol: str, exchange: str | None = None) -> dict:
        client = self._client(exchange or settings.default_exchange)
        return await client.fetch_ticker(symbol)

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()


market_data = MarketData()
