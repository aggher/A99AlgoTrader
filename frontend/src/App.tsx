import { useEffect, useState } from 'react'
import axios from 'axios'
import { Bot } from 'lucide-react'

import LivePrices        from './components/LivePrices'
import SignalCards        from './components/SignalCards'
import Chart              from './components/Chart'
import TradeHistory       from './components/TradeHistory'
import PerformanceMetrics from './components/PerformanceMetrics'
import AlertPanel         from './components/AlertPanel'
import { useWebSocket }   from './hooks/useWebSocket'

import { PriceData, SignalData, ModelMetric, TabId } from './types'

const API_URL = import.meta.env.VITE_API_URL || ''
const WS_URL = import.meta.env.VITE_API_URL 
  ? import.meta.env.VITE_API_URL.replace('http', 'ws') + '/ws'
  : `ws://${location.host}/ws`

export default function App() {
  const [tab,     setTab]     = useState<TabId>('dashboard')
  const [prices,  setPrices]  = useState<PriceData[]>([])
  const [signals, setSignals] = useState<SignalData[]>([])
  const [metrics, setMetrics] = useState<ModelMetric[]>([])
  const [tfFilter, setTfFilter] = useState<string>('ALL')

  const { connected, alerts, dismiss } = useWebSocket(WS_URL)

  // Poll REST endpoints
  useEffect(() => {
    const load = () => {
      axios.get(`${API_URL}/api/prices`).then(r  => setPrices(r.data.prices   ?? []))
      axios.get(`${API_URL}/api/signals`).then(r => setSignals(r.data.signals  ?? []))
      axios.get(`${API_URL}/api/performance`).then(r => setMetrics(r.data.metrics ?? []))
    }
    load()
    const id = setInterval(load, 15_000)
    return () => clearInterval(id)
  }, [])

  const filteredSignals = tfFilter === 'ALL' 
    ? signals 
    : signals.filter(s => s.timeframe === tfFilter)

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">🤖</div>
          <span className="logo-text">AlgoTrader AI</span>
        </div>
        <nav className="nav">
          {(['dashboard','signals','backtest','performance'] as TabId[]).map(t => (
            <button key={t} className={`nav-btn${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className="header-right">
          <div className="connection-badge">
            <div className={`dot${connected ? '' : ' off'}`}/>
            {connected ? 'Live' : 'Connecting…'}
          </div>
          <div className="connection-badge">
            <Bot size={13}/>
            <span>{signals.filter(s => s.signal !== 'HOLD').length} signals active</span>
          </div>
        </div>
      </header>

      {/* Alert toasts */}
      <AlertPanel alerts={alerts} dismiss={dismiss}/>

      {/* Main content */}
      <main className="page">
        {tab === 'dashboard' && (
          <>
            <LivePrices prices={prices}/>
            <div className="filter-row">
              <span className="text-muted">Filter:</span>
              {['ALL', '1m', '5m', '15m', '1h', '1d', '1mo'].map(f => (
                <button key={f} 
                  className={`filter-btn ${tfFilter === f ? 'active' : ''}`}
                  onClick={() => setTfFilter(f)}>
                  {f === 'ALL' ? 'ALL' : f}
                </button>
              ))}
            </div>
            <Chart/>
          </>
        )}

        {tab === 'signals' && (
          <>
            <div className="filter-row">
              <span className="text-muted">Filter:</span>
              {['ALL', '1m', '5m', '15m', '1h', '1d', '1mo'].map(f => (
                <button key={f} 
                  className={`filter-btn ${tfFilter === f ? 'active' : ''}`}
                  onClick={() => setTfFilter(f)}>
                  {f === 'ALL' ? 'ALL' : f}
                </button>
              ))}
            </div>
            <SignalCards signals={filteredSignals}/>
            <TradeHistory signals={filteredSignals}/>
          </>
        )}

        {tab === 'backtest' && (
          <PerformanceMetrics metrics={metrics}/>
        )}

        {tab === 'performance' && (
          <>
            <div className="card span-3">
              <div className="card-header">
                <span className="card-title">📊 Model Performance Summary</span>
              </div>
              <div className="metrics-grid">
                <div className="metric">
                  <div className="metric-lbl">Avg Accuracy</div>
                  <div className="metric-val b">
                    {metrics.length > 0
                      ? ((metrics.reduce((a,m) => a + (m.accuracy||0), 0) / metrics.length) * 100).toFixed(1) + '%'
                      : '—'}
                  </div>
                  <div className="metric-sub">across all models</div>
                </div>
                <div className="metric">
                  <div className="metric-lbl">Avg Sharpe</div>
                  <div className="metric-val g">
                    {metrics.length > 0
                      ? (metrics.reduce((a,m) => a + (m.sharpe_ratio||0), 0) / metrics.length).toFixed(2)
                      : '—'}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-lbl">Models Trained</div>
                  <div className="metric-val n">{metrics.length}</div>
                  <div className="metric-sub">symbol × timeframe</div>
                </div>
                <div className="metric">
                  <div className="metric-lbl">Avg F1 Score</div>
                  <div className="metric-val b">
                    {metrics.length > 0
                      ? ((metrics.reduce((a,m) => a + (m.f1_score||0), 0) / metrics.length) * 100).toFixed(1) + '%'
                      : '—'}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-lbl">Active Signals</div>
                  <div className="metric-val g">{signals.filter(s => s.signal !== 'HOLD').length}</div>
                </div>
                <div className="metric">
                  <div className="metric-lbl">Total Symbols</div>
                  <div className="metric-val n">10</div>
                  <div className="metric-sub">Core Institutional Pairs</div>
                </div>
              </div>
            </div>
            <PerformanceMetrics metrics={metrics}/>
          </>
        )}
      </main>
    </div>
  )
}
