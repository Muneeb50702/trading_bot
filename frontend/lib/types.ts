export type Action = "BUY" | "SELL" | "NO_TRADE";
export type Direction = "UP" | "DOWN" | "NEUTRAL";
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface Vote {
  module: string;
  bias: number;
  strength: number;
  note: string;
}

export interface Signal {
  symbol: string;
  timeframe: string;
  exchange: string;
  generated_at: string;
  direction: Direction;
  action: Action;
  confidence: number;
  probability_up: number;
  risk_level: RiskLevel;
  entry_price: number;
  stop_loss: number;
  take_profit_1: number;
  take_profit_2: number;
  take_profit_3: number;
  risk_reward: number;
  votes: Vote[];
  news_sentiment: string;
  rationale: string;
}

export interface PositionPlan {
  allowed: boolean;
  reason: string;
  position_size: number;
  quantity: number;
  risk_amount: number;
  risk_reward: number;
  leverage: number;
}

export interface SignalResult {
  signal: Signal;
  position_plan: PositionPlan;
  ml_probability: number | null;
  news_score: number;
  news_headlines: string[];
  id?: number;
}

export interface BacktestReport {
  symbol: string;
  timeframe: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_r: number;
  avg_r: number;
  profit_factor: number;
  max_drawdown_r: number;
  final_equity_r: number;
  equity_curve: number[];
}
