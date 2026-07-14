"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { SignalResult } from "@/lib/types";
import SignalCard from "@/components/SignalCard";
import BacktestPanel from "@/components/BacktestPanel";

const SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"];
const TIMEFRAMES = ["3m", "5m", "15m", "1h"];

export default function Dashboard() {
  const [results, setResults] = useState<SignalResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [tf, setTf] = useState("15m");
  const [live, setLive] = useState(false);
  const [model, setModel] = useState<{ trained: boolean } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try {
      const [scan, ms] = await Promise.all([
        api.scan(SYMBOLS.join(","), tf),
        api.modelStatus().catch(() => null),
      ]);
      setResults(scan.signals);
      setModel(ms);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, [tf]);

  useEffect(() => { load(); }, [load]);

  // live WebSocket feed
  useEffect(() => {
    if (!live) { wsRef.current?.close(); return; }
    const ws = new WebSocket(api.wsUrl());
    wsRef.current = ws;
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "signal" && msg.data?.signal) {
          setResults((prev) => {
            const d = msg.data as SignalResult;
            const key = (r: SignalResult) => `${r.signal.symbol}-${r.signal.timeframe}`;
            const map = new Map(prev.map((r) => [key(r), r]));
            map.set(key(d), d);
            return [...map.values()];
          });
        }
      } catch {}
    };
    return () => ws.close();
  }, [live]);

  const stats = useMemo(() => {
    const s = results.map((r) => r.signal);
    return {
      buys: s.filter((x) => x.action === "BUY").length,
      sells: s.filter((x) => x.action === "SELL").length,
      avgConf: s.length ? s.reduce((a, x) => a + x.confidence, 0) / s.length : 0,
    };
  }, [results]);

  const visible = results.filter((r) => r.signal.timeframe === tf);

  return (
    <main className="max-w-7xl mx-auto p-6">
      <header className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold">AI Futures Trading Bot</h1>
          <p className="text-muted text-sm">Probability-based signals · confidence scored · not financial advice</p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className={`px-2 py-1 rounded border ${model?.trained ? "border-up/40 text-up" : "border-edge text-muted"}`}>
            ML {model?.trained ? "active" : "confluence-only"}
          </span>
          <button onClick={() => setLive(!live)}
            className={`px-3 py-1 rounded border ${live ? "border-up/40 text-up bg-up/10" : "border-edge text-muted"}`}>
            {live ? "● LIVE" : "○ live off"}
          </button>
          <button onClick={load} className="px-3 py-1 rounded border border-accent/40 text-accent bg-accent/10">
            refresh
          </button>
        </div>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <StatTile label="Signals" value={`${visible.length}`} />
        <StatTile label="Buy" value={`${stats.buys}`} color="text-up" />
        <StatTile label="Sell" value={`${stats.sells}`} color="text-down" />
        <StatTile label="Avg Confidence" value={`${(stats.avgConf * 100).toFixed(0)}%`} />
      </div>

      <div className="flex items-center gap-2 mb-4">
        {TIMEFRAMES.map((t) => (
          <button key={t} onClick={() => setTf(t)}
            className={`px-3 py-1 rounded text-sm border ${tf === t ? "border-accent text-accent bg-accent/10" : "border-edge text-muted"}`}>
            {t}
          </button>
        ))}
      </div>

      {err && <div className="text-down bg-down/10 border border-down/30 rounded p-3 mb-4 text-sm">
        {err} — is the API running at {api.base}?
      </div>}

      {loading ? (
        <div className="text-muted">Loading signals…</div>
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {visible.map((r) => <SignalCard key={`${r.signal.symbol}-${r.signal.timeframe}`} result={r} />)}
          {visible.length === 0 && <div className="text-muted">No signals for {tf}.</div>}
        </div>
      )}

      <BacktestPanel symbol="BTC/USDT" timeframe={tf} />
    </main>
  );
}

function StatTile({ label, value, color = "text-gray-100" }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-panel border border-edge rounded-xl p-4">
      <div className={`text-2xl font-bold font-mono ${color}`}>{value}</div>
      <div className="text-xs text-muted uppercase tracking-wide mt-1">{label}</div>
    </div>
  );
}
