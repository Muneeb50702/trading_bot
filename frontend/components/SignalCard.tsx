"use client";
import { useState } from "react";
import type { SignalResult } from "@/lib/types";

const fmt = (n: number) => (n >= 1000 ? n.toLocaleString(undefined, { maximumFractionDigits: 2 }) : n.toPrecision(5));

function Badge({ action }: { action: string }) {
  const map: Record<string, string> = {
    BUY: "bg-up/15 text-up border-up/40",
    SELL: "bg-down/15 text-down border-down/40",
    NO_TRADE: "bg-muted/10 text-muted border-edge",
  };
  return <span className={`px-2 py-0.5 rounded text-xs font-bold border ${map[action]}`}>{action.replace("_", " ")}</span>;
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full bg-edge rounded overflow-hidden">
      <div className={color} style={{ width: `${Math.min(100, value * 100)}%`, height: "100%" }} />
    </div>
  );
}

export default function SignalCard({ result }: { result: SignalResult }) {
  const { signal: s, position_plan: plan, ml_probability, news_headlines } = result;
  const [open, setOpen] = useState(false);
  const up = s.direction === "UP";
  const riskColor = { LOW: "text-up", MEDIUM: "text-yellow-400", HIGH: "text-down" }[s.risk_level];

  return (
    <div className="bg-panel border border-edge rounded-xl p-4 flex flex-col gap-3 hover:border-accent/50 transition">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-bold text-lg">{s.symbol}</span>
          <span className="text-xs text-muted bg-panel2 px-1.5 py-0.5 rounded">{s.timeframe}</span>
        </div>
        <Badge action={s.action} />
      </div>

      <div className="flex items-center justify-between text-sm">
        <span className={up ? "text-up" : "text-down"}>{up ? "▲ UP" : "▼ DOWN"}</span>
        <span className="text-muted">P(up) {(s.probability_up * 100).toFixed(0)}%</span>
        <span className={riskColor}>{s.risk_level} risk</span>
      </div>

      <div>
        <div className="flex justify-between text-xs text-muted mb-1">
          <span>Confidence</span>
          <span>{(s.confidence * 100).toFixed(0)}%</span>
        </div>
        <Bar value={s.confidence} color={up ? "bg-up" : "bg-down"} />
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm font-mono">
        <Row label="Entry" value={fmt(s.entry_price)} />
        <Row label="SL" value={fmt(s.stop_loss)} className="text-down" />
        <Row label="TP1" value={fmt(s.take_profit_1)} className="text-up" />
        <Row label="TP2" value={fmt(s.take_profit_2)} className="text-up" />
        <Row label="TP3" value={fmt(s.take_profit_3)} className="text-up" />
        <Row label="R:R" value={`${s.risk_reward}`} />
      </div>

      <div className="flex items-center justify-between text-xs text-muted">
        <span>News: <span className="text-gray-300">{s.news_sentiment}</span></span>
        {ml_probability != null && <span>ML {(ml_probability * 100).toFixed(0)}%</span>}
        <button onClick={() => setOpen(!open)} className="text-accent hover:underline">
          {open ? "hide" : "details"}
        </button>
      </div>

      {open && (
        <div className="border-t border-edge pt-3 flex flex-col gap-2 text-xs">
          {plan.allowed ? (
            <div className="text-muted">
              Size <span className="text-gray-200">${plan.position_size}</span> ·
              Qty <span className="text-gray-200">{plan.quantity}</span> ·
              Risk <span className="text-down">${plan.risk_amount}</span> ·
              Lev <span className="text-gray-200">{plan.leverage}x</span>
            </div>
          ) : (
            <div className="text-yellow-400">Position blocked: {plan.reason}</div>
          )}
          <div className="text-muted">Top drivers:</div>
          <div className="flex flex-col gap-1">
            {[...s.votes].sort((a, b) => b.strength - a.strength).slice(0, 6).map((v) => (
              <div key={v.module} className="flex items-center gap-2">
                <span className="w-32 truncate text-gray-300">{v.module}</span>
                <div className="flex-1"><Bar value={Math.abs(v.bias) * v.strength} color={v.bias >= 0 ? "bg-up" : "bg-down"} /></div>
                <span className="text-muted w-40 truncate">{v.note}</span>
              </div>
            ))}
          </div>
          {news_headlines?.length > 0 && (
            <div className="text-muted">
              <div className="mt-1">Headlines:</div>
              {news_headlines.slice(0, 3).map((h, i) => <div key={i} className="text-gray-400 truncate">• {h}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ label, value, className = "" }: { label: string; value: string; className?: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted">{label}</span>
      <span className={className}>{value}</span>
    </div>
  );
}
