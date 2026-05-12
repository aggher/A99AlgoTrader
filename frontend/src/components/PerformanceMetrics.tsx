import { useState } from 'react'
import { TrendingUp } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import axios from 'axios'
import { ModelMetric, BacktestResult } from '../types'

interface Props { metrics: ModelMetric[] }

const ALL_SYMBOLS = [
  'EURUSD', 'GBPUSD', 'GBPAUD', 'EURAUD', 'GBPNZD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURJPY', 'XAUUSD'
]

function pct(v: number|null|undefined): string {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}
function num(v: number|null|undefined, d = 2): string {
  if (v == null) return '—'; return v.toFixed(d)
}

export default function PerformanceMetrics({ metrics }: Props) {
  const [bt, setBt]         = useState<BacktestResult|null>(null)
  const [btSym, setBtSym]   = useState('EURUSD')
  const [btTf,  setBtTf]    = useState('1h')
  const [loading, setLoading]= useState(false)

  const latest = metrics.slice(0, 6)

  const runBacktest = () => {
    setLoading(true)
    const API_URL = import.meta.env.VITE_API_URL || '';
    axios.get(`${API_URL}/api/backtest/${btSym}?timeframe=${btTf}`)
      .then(r => setBt(r.data))
      .catch(() => setBt(null))
      .finally(() => setLoading(false))
  }

  const eqData = bt?.equity_curve.map((v, i) => ({ i, equity: v })) ?? []

  return (
    <>
      {/* Model metrics */}
      <div className="card span-3">
        <div className="card-header">
          <span className="card-title"><TrendingUp size={15}/>Model Performance</span>
          <span className="text-muted">{latest.length} models</span>
        </div>
        {latest.length === 0
          ? <div className="empty">No metrics. Run training first.</div>
          : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th><th>TF</th><th>Accuracy</th><th>Precision</th>
                    <th>Recall</th><th>F1</th><th>Sharpe</th><th>MaxDD</th><th>PF</th>
                  </tr>
                </thead>
                <tbody>
                  {latest.map((m,i) => (
                    <tr key={i}>
                      <td className="td-mono">{m.symbol}</td>
                      <td className="text-muted">{m.timeframe}</td>
                      <td className={m.accuracy >= 0.65 ? 'td-pos' : 'text-muted'}>{pct(m.accuracy)}</td>
                      <td className="text-muted">{pct(m.precision)}</td>
                      <td className="text-muted">{pct(m.recall)}</td>
                      <td className={m.f1_score >= 0.60 ? 'td-pos' : 'text-muted'}>{pct(m.f1_score)}</td>
                      <td className={m.sharpe_ratio >= 1 ? 'td-pos' : 'td-neg'}>{num(m.sharpe_ratio)}</td>
                      <td className="td-neg">{pct(m.max_drawdown)}</td>
                      <td className={m.profit_factor >= 1.5 ? 'td-pos' : 'text-muted'}>{num(m.profit_factor)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </div>

      {/* Backtest */}
      <div className="card span-3">
        <div className="card-header">
          <span className="card-title">📊 Backtest</span>
          <div className="flex items-center gap-2">
            <select className="ctrl" value={btSym} onChange={e => setBtSym(e.target.value)}>
              {ALL_SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select className="ctrl" value={btTf} onChange={e => setBtTf(e.target.value)}>
              {['1mo', '1d', '1h', '15m', '5m'].map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <button className="btn btn-primary" onClick={runBacktest} disabled={loading}>
              {loading ? '…' : 'Run'}
            </button>
          </div>
        </div>

        {bt && (
          <>
            <div className="metrics-grid" style={{ marginBottom:16 }}>
              <div className="metric"><div className="metric-lbl">Win Rate</div><div className={`metric-val ${bt.win_rate >= 0.55 ? 'g' : 'r'}`}>{pct(bt.win_rate)}</div><div className="metric-sub">{bt.wins}W / {bt.losses}L</div></div>
              <div className="metric"><div className="metric-lbl">Profit Factor</div><div className={`metric-val ${bt.profit_factor >= 1.5 ? 'g' : 'r'}`}>{num(bt.profit_factor)}</div></div>
              <div className="metric"><div className="metric-lbl">Sharpe</div><div className={`metric-val ${bt.sharpe_ratio >= 1 ? 'g' : 'r'}`}>{num(bt.sharpe_ratio)}</div></div>
              <div className="metric"><div className="metric-lbl">Max Drawdown</div><div className={`metric-val ${Math.abs(bt.max_drawdown) < 0.15 ? 'g' : 'r'}`}>{pct(bt.max_drawdown)}</div></div>
              <div className="metric"><div className="metric-lbl">Total Return</div><div className={`metric-val ${bt.total_return >= 0 ? 'g' : 'r'}`}>{pct(bt.total_return)}</div></div>
              <div className="metric"><div className="metric-lbl">Trades</div><div className="metric-val n">{bt.total_trades}</div></div>
            </div>

            <div className="equity-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={eqData} margin={{ top:4, right:4, left:0, bottom:0 }}>
                  <defs>
                    <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#388bfd" stopOpacity={0.35}/>
                      <stop offset="95%" stopColor="#388bfd" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a2740"/>
                  <XAxis dataKey="i" hide />
                  <YAxis tickFormatter={v => `$${v.toLocaleString()}`} tick={{ fontSize:10, fill:'#8b949e' }} width={72} domain={['auto', 'auto']}/>
                  <Tooltip
                    contentStyle={{ background:'#101c2e', border:'1px solid #388bfd33', borderRadius:8, fontSize:11 }}
                    formatter={(v: number) => [`$${v.toLocaleString()}`, 'Equity']}
                  />
                  <Area type="monotone" dataKey="equity" stroke="#388bfd" fill="url(#eqGrad)" strokeWidth={2} dot={false} animationDuration={1000}/>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Detailed Trade Log */}
            <div className="card-header mt-4" style={{ marginTop: 24 }}>
              <span className="card-title">📜 Historical Execution Log</span>
              <span className="text-muted">Last 200 trades</span>
            </div>
            <div className="table-wrap" style={{ maxHeight: 300, overflowY: 'auto' }}>
              <table className="table-small">
                <thead>
                  <tr>
                    <th>Time</th><th>Signal</th><th>Entry</th><th>Exit</th><th>Exit Reason</th><th>PnL</th><th>Conf</th>
                  </tr>
                </thead>
                <tbody>
                  {bt.trade_log.slice().reverse().map((t, i) => (
                    <tr key={i}>
                      <td className="td-mono text-xs">{t.time.split(' ')[0]}</td>
                      <td>
                        <span className={`badge ${t.signal === 'BUY' ? 'badge-buy' : 'badge-sell'}`}>
                          {t.signal}
                        </span>
                      </td>
                      <td className="td-mono">{t.entry.toFixed(5)}</td>
                      <td className="td-mono">{t.exit.toFixed(5)}</td>
                      <td>
                        <span className={`text-xs ${t.reason === 'STOP_LOSS' ? 'r' : t.reason === 'TAKE_PROFIT' ? 'g' : 'text-muted'}`}>
                          {t.reason.replace('_', ' ')}
                        </span>
                      </td>
                      <td className={`td-mono ${t.pnl >= 0 ? 'g' : 'r'}`}>
                        {t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)}
                      </td>
                      <td className="text-muted text-xs">{(t.confidence * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        {!bt && !loading && <div className="empty">Select a symbol and run backtest to see equity curve.</div>}
      </div>
    </>
  )
}
