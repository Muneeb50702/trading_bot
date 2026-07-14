# AI Futures Trading Bot

A probability-based crypto **futures** signal engine. It ingests live market data,
runs a large bank of technical + Smart-Money analysis modules, fuses them into a
calibrated probability, refines that with a machine-learning model, and emits
actionable signals — **Direction, Entry, Stop-Loss, TP1/TP2/TP3, Confidence, Risk
Level, and Buy/Sell/No-Trade** — gated by real risk management.

> ⚠️ **Not financial advice.** This tool produces *probability-based* predictions with
> confidence scores. It deliberately never implies certainty or guarantees accuracy or
> profit. Trade at your own risk.

---

## Why it's built this way

| Requirement | Decision |
|---|---|
| 5 exchanges (Binance/Bybit/OKX/Bitget/BingX) | One unified async **ccxt** client — no five SDKs |
| AI/ML that's honestly probabilistic | **LightGBM** gradient-boosted trees on tabular indicator features: trains in seconds, calibrated probabilities, <1ms inference. Pluggable for a deep model later |
| "No 90% guarantee, confidence-based" | Gentle logistic calibration caps confidence well short of certainty; ML blended, not trusted blindly |
| Works before any training | Deterministic **confluence engine** produces signals on day one; ML *refines* when a model exists |
| 1–3 s, low latency, scalable | Fully async FastAPI; vectorized NumPy indicators; concurrent symbol×timeframe scans |
| Runs with zero infra to try | **SQLite fallback**; docker-compose brings the full Postgres + Redis stack |

---

## Architecture

```
                         ┌──────────────── Next.js dashboard ────────────────┐
                         │  live signal grid · confidence bars · backtest UI │
                         └───────────────▲───────────────────▲───────────────┘
                              REST /api  │        WebSocket   │ /ws/signals
┌────────────────────────────────────────┴────────────────────┴──────────────┐
│                            FastAPI application                              │
│  auth+2FA+RBAC · signals · backtest · model · admin/analytics · settings    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Signal orchestration pipeline (app/signals/service.py)                     │
│    ccxt OHLCV+OI ─► confluence ─► ML blend ─► news ─► signal builder ─► risk │
│         │              │             │          │            │          │    │
│   data/market.py   analysis/**   ml/model.py  news/**  engine/signal_  risk/ │
│   (Binance…BingX)  21 modules   (LightGBM)   sentiment  builder.py    manager│
├─────────────────────────────────────────────────────────────────────────────┤
│  Persistence: SQLAlchemy (SQLite ▸ Postgres) · Redis · notifications        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Analysis modules (each casts a weighted bias vote → confluence)
**Trend:** EMA stack · SMA cross · ADX/DI · SuperTrend · Ichimoku
**Momentum:** RSI · MACD · Stochastic RSI · Stochastic
**Volatility:** Bollinger %B/width · ATR regime
**Volume:** VWAP · OBV · Volume Profile (POC)
**Price action:** Market structure (HH/HL) · Support/Resistance · Candlestick patterns
**Smart Money Concepts:** BOS/CHOCH · Order Blocks · Fair Value Gaps · Open Interest

---

## Quick start

### Option A — Docker (full stack: Postgres + Redis + API + dashboard)
```bash
cp .env.example .env        # edit BOT_JWT_SECRET etc.
docker compose up --build
# dashboard  → http://localhost:3000
# API + docs → http://localhost:8000/docs
```

### Option B — Local dev (zero infra, SQLite)
```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload            # http://localhost:8000/docs

# frontend (new terminal)
cd frontend
npm install
npm run dev                              # http://localhost:3000
```

Enable the always-on background scanner (persists + broadcasts + notifies):
```bash
BOT_ENABLE_SCANNER=1 uvicorn app.main:app
```

Run the offline engine test (no network needed):
```bash
cd backend && source .venv/bin/activate && python scripts/smoke_test.py
```

---

## Key API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/register` · `/login` | Auth (first user = admin); optional TOTP 2FA |
| GET | `/api/signals/generate?symbol=BTC/USDT&timeframe=15m` | One signal, full detail |
| GET | `/api/signals/scan?symbols=…&timeframes=…` | Concurrent grid scan |
| GET | `/api/signals/history` | Stored signal history |
| GET | `/api/backtest/run?symbol=…&timeframe=…&candles=1000` | Win rate, P/L, PF, drawdown, equity curve |
| GET | `/api/model/status` · POST `/api/model/train` | ML model inspect / retrain (admin) |
| GET | `/api/admin/health` · `/analytics` · `/users` · `/audit` | Ops & analytics |
| GET/PUT | `/api/settings` · POST `/api/settings/api-keys` | Per-user config; encrypted exchange keys |
| WS | `/ws/signals` | Live signal push |

Full interactive docs at **`/docs`**.

---

## Signal anatomy

```jsonc
{
  "signal": {
    "symbol": "BTC/USDT", "timeframe": "15m", "direction": "UP",
    "action": "BUY", "confidence": 0.70, "probability_up": 0.76,
    "risk_level": "MEDIUM",
    "entry_price": 63787.7, "stop_loss": 63516.3,
    "take_profit_1": 64059.1, "take_profit_2": 64330.5, "take_profit_3": 64601.9,
    "risk_reward": 1.0, "news_sentiment": "NEUTRAL",
    "votes": [ /* every module's bias + strength + note, for full transparency */ ],
    "rationale": "P(up)=76%, UP @ 70% conf. Drivers: supertrend(...), bos_choch(...)"
  },
  "position_plan": { "allowed": true, "position_size": 1275.7, "quantity": 0.02,
                     "risk_amount": 100.0, "leverage": 5 },
  "ml_probability": 0.74, "news_score": 0.0
}
```

---

## Security
- Passwords hashed with **bcrypt**; JWT sessions; **RBAC** (admin/user); optional **TOTP 2FA**.
- Exchange API keys **encrypted at rest** (Fernet) and never returned by the API.
- **Audit log** of auth/security events.

## Configuration
All backend settings are env vars prefixed `BOT_` (see [.env.example](.env.example) and
[backend/app/core/config.py](backend/app/core/config.py)) — thresholds, risk limits,
ATR multipliers, TP R-multiples, scan cadence, notification credentials.

## Project layout
```
backend/app/
  analysis/     21 vote-casting modules (trend, momentum, volatility, volume, price action, SMC)
  ta/core.py    vectorized indicator math (no TA-Lib)
  engine/       confluence fusion + signal builder
  ml/           feature engineering + LightGBM model + predictor/blender
  data/ news/ risk/            market data · sentiment · risk manager
  signals/      orchestration service + background scanner
  backtest/     event-driven backtester
  api/          routers (auth, signals, backtest, model, admin, settings)
  db/ core/     ORM models + session · config, logging, security
frontend/       Next.js 14 + Tailwind dashboard (live grid, confidence bars, backtest)
docker-compose.yml
```

## Roadmap hooks (already wired, ready to extend)
- Swap LightGBM for a PyTorch/TF sequence model behind `app/ml/model.py`'s interface.
- WhatsApp/push via the same `Notifier` channel pattern.
- Live order execution using the already-encrypted per-user exchange keys.
