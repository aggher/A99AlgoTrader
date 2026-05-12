export interface PriceData {
  symbol: string; close: number; open: number; high: number; low: number
  volume: number; timestamp: string
}
export interface SignalData {
  id?: number; symbol: string; timeframe: string; timestamp: string
  signal: 'BUY'|'SELL'|'HOLD'; probability: number
  entry: number|null; stop_loss: number|null; take_profit: number|null
  rsi_divergence: number; volume_spike: number; volatility_breakout: number
  position_size?: number|null; confluence_score?: number
  meta_score?: number; meta_decision?: string
  market_regime?: string; regime_strength?: number
  agreement?: number; confidence?: number
  // Live confluence fields from feature engine
  rsi?: number; atr?: number; bb_width?: number
  prob_buy?: number; prob_sell?: number; prob_hold?: number
}
export interface OHLCVBar {
  time: number; open: number; high: number; low: number; close: number; volume: number
}
export interface ModelMetric {
  symbol: string; timeframe: string; trained_at: string
  accuracy: number; precision: number; recall: number; f1_score: number
  sharpe_ratio: number; max_drawdown: number; profit_factor: number
  params: Record<string,unknown>
}
export interface BacktestResult {
  symbol: string; timeframe: string; total_trades: number
  wins: number; losses: number; win_rate: number; profit_factor: number
  sharpe_ratio: number; max_drawdown: number; total_return: number
  equity_curve: number[]; trade_log: TradeRecord[]
}
export interface TradeRecord {
  entry_time: string; exit_time: string; direction: 'BUY'|'SELL'
  entry_price: number; exit_price: number; pnl: number
  outcome: 'take_profit'|'stop_loss'|'timeout'; probability: number
}
export type TabId = 'dashboard'|'signals'|'backtest'|'performance'
