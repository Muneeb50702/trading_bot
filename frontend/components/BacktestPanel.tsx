"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import type { BacktestReport } from "@/lib/types";

export default function BacktestPanel({ symbol, timeframe }: { symbol: string; timeframe: string }) {
  const [report, setReport] = useState<BacktestReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setErr(null);
    try {
      setReport(await api.backtest(symbol, timeframe, 1000));
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  };

  return (
    <div className="bg-panel border border-edge rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">Backtest · {symbol} {timeframe}</h3>
        <button onClick={run} disabled={loading}
          className="text-xs bg-accent/20 text-accent border border-accent/40 rounded px-3 py-1 hover:bg-accent/30 disabled:opacity-50">
          {loading ? "running…" : "run backtest"}
        </button>
      </div>
      {err && <div className="text-down text-xs">{err}</div>}
      {report && (
        <>
          <div className="grid grid-cols-3 gap-3 text-center">
            <Stat label="Trades" value={`${report.trades}`} />
            <Stat label="Win Rate" value={`${(report.win_rate * 100).toFixed(1)}%`} accent={report.win_rate >= 0.5} />
            <Stat label="Total R" value={`${report.total_r}`} accent={report.total_r >= 0} />
            <Stat label="Avg R" value={`${report.avg_r}`} accent={report.avg_r >= 0} />
            <Stat label="Profit Factor" value={`${report.profit_factor}`} accent={report.profit_factor >= 1} />
            <Stat label="Max DD" value={`${report.max_drawdown_r}R`} accent={false} />
          </div>
          <Equity curve={report.equity_curve} />
        </>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  const color = accent === undefined ? "text-gray-100" : accent ? "text-up" : "text-down";
  return (
    <div className="bg-panel2 rounded-lg py-2">
      <div className={`font-mono font-bold ${color}`}>{value}</div>
      <div className="text-[10px] text-muted uppercase tracking-wide">{label}</div>
    </div>
  );
}

function Equity({ curve }: { curve: number[] }) {
  if (!curve.length) return null;
  const w = 100, h = 32;
  const min = Math.min(0, ...curve), max = Math.max(0, ...curve);
  const span = max - min || 1;
  const pts = curve.map((v, i) => `${(i / (curve.length - 1)) * w},${h - ((v - min) / span) * h}`).join(" ");
  const last = curve[curve.length - 1];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16 mt-3" preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={last >= 0 ? "#16c784" : "#ea3943"} strokeWidth="0.7" />
    </svg>
  );
}
