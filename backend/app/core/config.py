"""Central configuration, loaded from environment / .env.

Everything tunable about the bot lives here so behaviour can be changed
without touching engine code.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BOT_", extra="ignore")

    # --- App ---
    app_name: str = "AI Futures Trading Bot"
    env: str = "dev"
    log_level: str = "INFO"

    # --- Persistence ---
    # Defaults to a local SQLite file so the API runs with zero infra.
    # docker-compose overrides this with the Postgres DSN.
    database_url: str = "sqlite+aiosqlite:///./trading_bot.db"
    redis_url: str = "redis://localhost:6379/0"

    # --- Security ---
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24
    # Fernet key used to encrypt stored exchange API keys at rest.
    encryption_key: str = ""

    # --- Exchange / market data ---
    default_exchange: str = "binanceusdm"  # ccxt id for Binance USD-M futures
    default_symbols: list[str] = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
    ]
    default_timeframes: list[str] = ["3m", "5m", "15m", "1h"]
    ohlcv_limit: int = 500  # candles pulled per analysis pass

    # --- Signal engine ---
    # Probability thresholds that turn a raw prob into Buy / Sell / No-Trade.
    long_threshold: float = 0.58
    short_threshold: float = 0.42
    min_confidence: float = 0.55  # below this -> No Trade regardless of side
    # ML blend weight: final_prob = w*ml + (1-w)*confluence, when a model exists.
    ml_blend_weight: float = 0.5

    # --- Risk management (defaults; per-user overridable) ---
    account_balance: float = 10_000.0
    risk_per_trade_pct: float = 1.0        # % of balance risked per trade
    max_daily_loss_pct: float = 5.0        # halt trading after this daily drawdown
    max_consecutive_losses: int = 4
    default_leverage: int = 5
    atr_sl_mult: float = 1.5               # SL distance = mult * ATR
    tp_r_multiples: list[float] = [1.0, 2.0, 3.0]  # TP1/2/3 in R multiples

    # --- Notifications ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # --- Live loop ---
    scan_interval_seconds: int = 30  # background scan cadence per symbol/timeframe


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
