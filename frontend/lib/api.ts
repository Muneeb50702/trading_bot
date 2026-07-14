import type { BacktestReport, SignalResult } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  base: API,
  scan: (symbols?: string, timeframes?: string) => {
    const q = new URLSearchParams();
    if (symbols) q.set("symbols", symbols);
    if (timeframes) q.set("timeframes", timeframes);
    return get<{ count: number; signals: SignalResult[] }>(`/api/signals/scan?${q}`);
  },
  generate: (symbol: string, timeframe: string) =>
    get<SignalResult>(`/api/signals/generate?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&persist=false`),
  backtest: (symbol: string, timeframe: string, candles = 1000) =>
    get<BacktestReport>(`/api/backtest/run?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&candles=${candles}`),
  modelStatus: () => get<{ trained: boolean; meta: Record<string, unknown>; blend_weight: number }>(`/api/model/status`),
  health: () => get<Record<string, unknown>>(`/api/admin/health`),
  wsUrl: () => `${API.replace(/^http/, "ws")}/ws/signals`,
};
