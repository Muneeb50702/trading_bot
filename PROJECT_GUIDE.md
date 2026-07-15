# AI Futures Trading Bot — Complete Project Guide

A plain-language, head-to-toe explanation of **what this project is, how every piece
works, and how it all fits together**. Read this top to bottom and you'll understand
the whole system well enough to explain it, demo it, and defend it to a client.

---

## 1. The one-paragraph summary

This is a system that **watches cryptocurrency futures markets live, analyzes them with
~21 different professional trading techniques at once, combines all those opinions into a
single probability, sharpens that probability with a machine-learning model, and then
prints a clear trade idea** — *should you go long or short, at what price, where to put
your stop-loss, three take-profit targets, how confident it is, and how risky the trade
is.* It never promises certainty; it speaks in probabilities and confidence scores, which
is exactly how professional quant systems talk. It comes with a web dashboard, a secure
API, user accounts, a backtester to check historical performance, and one-command Docker
deployment.

---

## 2. The big picture (how the parts connect)

```
   LIVE MARKET (Binance, Bybit, OKX, Bitget, BingX)
              │  price candles + open interest
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 1. DATA LAYER      pulls OHLCV candles via ccxt       │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 2. ANALYSIS BANK   21 modules each give a "vote"      │
   │    (RSI, MACD, SuperTrend, Order Blocks, FVG, …)      │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 3. CONFLUENCE      blends all votes → a probability   │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 4. ML MODEL        LightGBM refines the probability   │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 5. NEWS            bullish/bearish/neutral nudge       │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 6. SIGNAL BUILDER  → Direction, Entry, SL, TP1-3,     │
   │                      Confidence, Risk, Buy/Sell/No    │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 7. RISK MANAGER    position size, daily-loss limits   │
   └─────────────────────────────────────────────────────┘
              ▼
   ┌───────────────┬──────────────┬─────────────────────┐
   │ Dashboard (UI)│ Database      │ Telegram / Email     │
   │ WebSocket live│ (history)     │ notifications        │
   └───────────────┴──────────────┴─────────────────────┘
```

**Think of it like a panel of 21 expert traders.** Each expert looks at the same chart
but with their own specialty. They each say "I lean bullish" or "I lean bearish" and how
strongly. A chairman (the confluence engine) tallies the weighted opinions. A statistician
(the ML model) adjusts the tally based on what actually happened in similar situations
historically. A news analyst adds context. Finally a risk officer decides the exact trade
and whether it's even safe to take right now.

---

## 3. Walkthrough: the life of one signal

Let's follow a single request — *"give me a signal for BTC/USDT on the 15-minute chart"*:

1. **Fetch data** — [backend/app/data/market.py](backend/app/data/market.py) asks Binance
   (through the ccxt library) for the last 500 fifteen-minute candles plus open-interest
   history. Results are briefly cached so rapid requests don't hammer the exchange.

2. **Run the analysis bank** — [backend/app/engine/confluence.py](backend/app/engine/confluence.py)
   runs all 21 modules in [backend/app/analysis/](backend/app/analysis/). Each returns a
   **Vote**: a `bias` from −1 (very bearish) to +1 (very bullish) and a `strength` from 0
   to 1 (how much to trust this vote right now).

3. **Blend the votes** — the confluence engine computes a weighted average of all votes
   (some modules are given more weight because they're structurally more reliable) and
   squashes it into a **probability that price goes up**, e.g. `P(up) = 0.76`. The squash
   is deliberately gentle so it never reaches extreme numbers like 99%.

4. **ML refinement** — if a trained model exists,
   [backend/app/ml/predictor.py](backend/app/ml/predictor.py) turns the current market
   into a row of ~22 numeric features and asks the LightGBM model for its own probability,
   then blends the two (50/50 by default). Untrained? It just uses the confluence number —
   the bot still works.

5. **News nudge** — [backend/app/news/sentiment.py](backend/app/news/sentiment.py) checks
   recent crypto headlines about BTC and returns Bullish/Bearish/Neutral, which slightly
   adjusts the confidence.

6. **Build the signal** — [backend/app/engine/signal_builder.py](backend/app/engine/signal_builder.py)
   turns the final probability into a concrete plan:
   - **Direction**: UP if P(up) ≥ 0.5, else DOWN.
   - **Action**: BUY if P(up) is high *and* confidence is high enough; SELL if low; else
     NO-TRADE (don't force a trade when the market is unclear).
   - **Entry** = current price. **Stop-Loss** = entry ± (1.5 × ATR), where ATR measures
     recent volatility, so the stop adapts to how choppy the market is.
   - **TP1/TP2/TP3** = 1×, 2×, 3× the risk distance (so TP1 is a 1:1 reward-to-risk target).
   - **Confidence** = combination of how far the probability is from a coin-flip, how much
     the 21 modules agree, and whether the news aligns.
   - **Risk Level** = LOW/MEDIUM/HIGH based on volatility, leverage, and confidence.

7. **Risk check** — [backend/app/risk/manager.py](backend/app/risk/manager.py) sizes the
   position so that if the stop-loss is hit you only lose your configured risk (e.g. 1% of
   balance), and it **blocks new trades** if you've hit your daily loss limit or too many
   losses in a row (the "kill switch").

8. **Deliver** — the result is saved to the database, pushed live to any open dashboard
   over WebSocket, and (for BUY/SELL) sent to Telegram/email.

---

## 4. Every layer explained

### 4.1 Data layer — `app/data/market.py`
Uses **ccxt**, a library that speaks to 100+ exchanges with one unified interface. That's
why supporting Binance, Bybit, OKX, Bitget, and BingX didn't require five separate
integrations. It fetches candles (OHLCV = Open, High, Low, Close, Volume) and, when the
exchange offers it, open-interest history (how many futures contracts are open — a
smart-money clue).

### 4.2 Indicator math — `app/ta/core.py`
Pure, fast mathematical functions built on NumPy/pandas (no heavy external TA library).
These are the building blocks: moving averages, RSI, MACD, ATR, ADX, Bollinger Bands,
SuperTrend, Ichimoku, VWAP, OBV, Volume Profile, Stochastics. Every analysis module is
built from these.

### 4.3 The 21 analysis modules — `app/analysis/`
Each file groups related modules. Every module reads the candles and returns one Vote.

| Group | Modules | What they read |
|---|---|---|
| **Trend** ([trend.py](backend/app/analysis/trend.py)) | EMA stack, SMA cross, ADX/DI, SuperTrend, Ichimoku | Is the market trending, and which way? |
| **Momentum** ([momentum.py](backend/app/analysis/momentum.py)) | RSI, MACD, Stochastic RSI, Stochastic | Is momentum building or fading? Overbought/oversold? |
| **Volatility** ([volatility.py](backend/app/analysis/volatility.py)) | Bollinger %B/width, ATR regime | Is the market expanding or squeezing? |
| **Volume** ([volume.py](backend/app/analysis/volume.py)) | VWAP, OBV, Volume Profile | Is volume confirming the move? Where did most trading happen? |
| **Price action** ([price_action.py](backend/app/analysis/price_action.py)) | Market structure (HH/HL), Support/Resistance, Candlestick patterns | Trend structure and reversal candles |
| **Smart Money Concepts** ([smc.py](backend/app/analysis/smc.py)) | BOS/CHOCH, Order Blocks, Fair Value Gaps, Open Interest | Where institutions likely bought/sold and whether trend continues or flips |

The design is **pluggable**: adding a 22nd technique is just writing one function with a
`@register` decorator — the confluence engine picks it up automatically.

### 4.4 Confluence engine — `app/engine/confluence.py`
The "chairman." Weighted-averages all votes and converts to a probability. Because it's
purely rule-based, **the bot produces sensible signals from day one, with no training
required.**

### 4.5 Machine learning — `app/ml/`
- [features.py](backend/app/ml/features.py) turns market state into ~22 numbers (returns,
  RSI, MACD histogram, trend alignment, volatility, etc.).
- [model.py](backend/app/ml/model.py) wraps **LightGBM**, a gradient-boosted decision-tree
  model. We chose it over TensorFlow/PyTorch because for this kind of tabular, numeric data
  it trains in seconds, gives honest calibrated probabilities (perfect for the "no 90%
  guarantee" requirement), and predicts in under a millisecond. The interface is generic,
  so a deep-learning model can replace it later without touching the rest of the system.
- It learns from history by labelling each past bar with "did price actually go up over the
  next few candles?" and only learns from decisive moves (filtering out noise).
- [predictor.py](backend/app/ml/predictor.py) blends the ML probability with the confluence
  probability.

### 4.6 Signal builder — `app/engine/signal_builder.py`
Converts probability → tradeable plan (covered in the walkthrough). This is where Entry,
Stop-Loss, three Take-Profits, Confidence, Risk Level, and the Buy/Sell/No-Trade decision
are computed.

### 4.7 Risk manager — `app/risk/manager.py`
Position sizing (risk a fixed % per trade), leverage cap, and the **kill switch** that
halts trading after a daily-loss limit or a losing streak. This is what separates a toy
from something that respects real capital.

### 4.8 Backtester — `app/backtest/engine.py`
Replays historical candles, generates a signal at each step, simulates the trade (enter
next candle, exit at stop or target), and reports **win rate, total profit in "R" units,
profit factor, max drawdown, and an equity curve.** This lets you sanity-check a strategy
before risking money.

### 4.9 API — `app/api/` + `app/main.py`
A **FastAPI** web server exposing everything over HTTP + WebSocket. Interactive docs are
auto-generated at `/docs`. Routers:
- `auth` — register/login, JWT tokens, 2FA
- `signals` — generate one, scan the whole grid, history
- `backtest` — run a backtest
- `model` — check status, retrain
- `admin` — health, analytics, users, audit log
- `settings` — user preferences, encrypted exchange keys

### 4.10 Background scanner — `app/signals/scanner.py`
An always-on loop (toggle with `BOT_ENABLE_SCANNER=1`) that re-scans every symbol/timeframe
on a timer, saves signals, pushes them live, and fires notifications. This is the "bot
running by itself" part.

### 4.11 Notifications — `app/notifications/`
Telegram and email out of the box, plus WebSocket push to the dashboard. Same pattern
extends to WhatsApp/push.

### 4.12 Database & security — `app/db/`, `app/core/security.py`
- Data stored via SQLAlchemy — **SQLite** by default (zero setup) or **PostgreSQL** in
  production. Tables: users, settings, exchange keys, signals, trades, audit log.
- Passwords hashed (bcrypt), sessions via JWT, roles (admin/user), optional TOTP 2FA
  (Google Authenticator), exchange API keys **encrypted at rest**, and an audit log of
  security events.

### 4.13 Frontend — `frontend/`
A **Next.js 14 + Tailwind** dashboard: live signal cards with confidence bars, the exact
Entry/SL/TP levels, a breakdown of which modules voted which way, a timeframe filter, a
"LIVE" toggle that streams updates over WebSocket, and a backtest panel with an equity
curve chart.

---

## 5. How to run it

**Try it instantly (no database needed):**
```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload      # open http://localhost:8000/docs
```
**Full stack in one command:**
```bash
cp .env.example .env               # set a JWT secret
docker compose up --build          # dashboard :3000, API :8000
```
**Prove the engine works offline:**
```bash
cd backend && source .venv/bin/activate && python scripts/smoke_test.py
```

---

## 6. Trading glossary (so the terms make sense)

- **Futures** — contracts to buy/sell an asset later; allow leverage and short-selling.
- **Long / Short** — betting price goes up / down.
- **OHLCV** — a candle's Open, High, Low, Close, Volume.
- **Entry / Stop-Loss / Take-Profit** — where you get in, your safety exit if wrong, and
  your profit targets.
- **ATR** — Average True Range; a volatility measure used to size stops.
- **R / Risk-Reward** — "R" is the amount risked; a 3R profit means you made 3× your risk.
- **Confidence** — how sure the system is; **not** a guarantee.
- **SMC (Smart Money Concepts)** — reading charts the way institutions move: Order Blocks
  (institutional buy/sell zones), Fair Value Gaps (price imbalances), BOS/CHOCH (trend
  continuation vs reversal).
- **Open Interest** — number of open futures contracts; rising OI + rising price = new
  money entering.
- **Backtest** — testing a strategy on historical data.
- **Drawdown** — the biggest drop from a peak; measures pain/risk.

---

## 7. What's production-ready vs. what needs more work

**Solid now:** the full analysis→signal pipeline, live multi-exchange data, backtesting,
API, auth/security, dashboard, Docker deployment, graceful degradation.

**Needs attention before real-money, live trading:**
- **News feed** currently uses a free endpoint that now needs an API key → plug in a keyed
  news/sentiment source to activate it (otherwise it stays Neutral).
- **The ML model ships untrained** — it must be trained on the client's chosen pairs, and
  ideally retrained on a schedule.
- **No live order execution yet** — it generates signals; wiring it to actually place
  orders (using the already-encrypted keys) is a deliberate, separate, higher-risk step.
- **Strategy tuning** — thresholds and module weights should be tuned/backtested per market.
- **Legal** — any trading tool needs a clear "not financial advice / trade at your own
  risk" disclaimer and a liability clause in the contract.

---

*This document is part of the project deliverables. For the engineering-focused overview,
see [README.md](README.md).*
