import { Zap, TrendingUp, TrendingDown, BarChart2, Activity } from 'lucide-react'
import { SignalData } from '../types'

interface Props { signals: SignalData[] }

function fmt(v: number|null|undefined, digits = 5): string {
  if (v == null) return '—'
  if (v >= 100) return v.toFixed(2)
  return v.toFixed(digits)
}

function rsiColor(rsi: number) {
  if (rsi >= 70) return 'down'
  if (rsi <= 30) return 'up'
  return 'neutral'
}

function ConfluenceMeter({ score, max = 5 }: { score: number; max?: number }) {
  return (
    <span className="conf-dots">
      {Array.from({ length: max }, (_, i) => (
        <span key={i} className={`conf-dot${i < score ? ' on' : ''}`} />
      ))}
    </span>
  )
}

function ProbBar({ probBuy, probHold, probSell }: { probBuy: number; probHold: number; probSell: number }) {
  return (
    <div style={{ display: 'flex', height: 6, borderRadius: 4, overflow: 'hidden', margin: '6px 0' }}>
      <div style={{ width: `${probBuy * 100}%`, background: 'var(--up)', opacity: 0.85 }} />
      <div style={{ width: `${probHold * 100}%`, background: 'var(--text-muted)', opacity: 0.4 }} />
      <div style={{ width: `${probSell * 100}%`, background: 'var(--down)', opacity: 0.85 }} />
    </div>
  )
}

function SignalCard({ s }: { s: SignalData }) {
  const cls  = s.signal.toLowerCase()
  const prob = Math.round(s.probability * 100)
  const conf = s.confluence_score ?? 0

  const probBuy  = s.prob_buy  ?? (s.signal === 'BUY'  ? s.probability : 0)
  const probSell = s.prob_sell ?? (s.signal === 'SELL' ? s.probability : 0)
  const probHold = s.prob_hold ?? (s.signal === 'HOLD' ? s.probability : 1 - s.probability)

  const rsi = s.rsi ?? 50
  const atr = s.atr ?? 0
  const bbW = s.bb_width ?? 0

  const isOverbought = rsi >= 70
  const isOversold   = rsi <= 30

  return (
    <div className={`sig-card ${cls}`}>
      {/* Header */}
      <div className="sig-head">
        <div>
          <div className="sig-sym">{s.symbol}</div>
          <div className="sig-tf">{s.timeframe}</div>
        </div>
        <span className={`sig-badge ${cls}`}>
          {s.signal === 'BUY' ? <TrendingUp size={11} style={{ marginRight: 3 }} /> :
           s.signal === 'SELL' ? <TrendingDown size={11} style={{ marginRight: 3 }} /> : null}
          {s.signal}
        </span>
      </div>

      {/* Probability bar — 3-way split */}
      <div className="sig-prob-row">
        <span className="text-muted">Model Confidence</span>
        <span className={`sig-prob-pct mono ${cls === 'buy' ? 'up' : cls === 'sell' ? 'down' : 'flat'}`}>
          {prob}%
        </span>
      </div>
      <ProbBar probBuy={probBuy} probHold={probHold} probSell={probSell} />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 8 }}>
        <span style={{ color: 'var(--up)' }}>B {Math.round(probBuy * 100)}%</span>
        <span>H {Math.round(probHold * 100)}%</span>
        <span style={{ color: 'var(--down)' }}>S {Math.round(probSell * 100)}%</span>
      </div>

      {/* Entry / SL / TP */}
      <div className="sig-levels">
        <div className="sig-level"><div className="level-lbl">Entry</div><div className="level-val">{fmt(s.entry)}</div></div>
        <div className="sig-level"><div className="level-lbl">Stop Loss</div><div className="level-val down">{fmt(s.stop_loss)}</div></div>
        <div className="sig-level"><div className="level-lbl">Take Profit</div><div className="level-val up">{fmt(s.take_profit)}</div></div>
      </div>

      {/* Live Technical Indicators */}
      <div className="sig-confluence" style={{ marginTop: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px 12px', marginBottom: 8 }}>
          <div className="conf-item">
            <span className="conf-label">RSI(14)</span>
            <span className={`conf-val ${rsiColor(rsi)}`}>{rsi.toFixed(1)}</span>
          </div>
          <div className="conf-item">
            <span className="conf-label">ATR(20)</span>
            <span className="conf-val neutral">{atr > 0 ? atr.toFixed(5) : '—'}</span>
          </div>
          <div className="conf-item">
            <span className="conf-label">BB Width</span>
            <span className="conf-val neutral">{bbW > 0 ? bbW.toFixed(4) : '—'}</span>
          </div>
          <div className="conf-item">
            <span className="conf-label">Regime</span>
            <span className="conf-val neutral">{s.market_regime || 'UNKNOWN'}</span>
          </div>
          <div className="conf-item">
            <span className="conf-label">Agreement</span>
            <span className={`conf-val ${(s.agreement ?? 0) > 0.8 ? 'up' : 'neutral'}`}>
              {Math.round((s.agreement ?? 0) * 100)}%
            </span>
          </div>
          <div className="conf-item">
            <span className="conf-label">Alpha</span>
            <span className={`conf-val ${(s.confidence ?? 0) > 0.6 ? 'up' : 'neutral'}`}>
              {((s.confidence ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Confluence Tags */}
        <div className="sig-tags">
          <span className={`tag ${s.rsi_divergence !== 0 ? 'on' : 'off'}`}>RSI Div</span>
          <span className={`tag ${s.volume_spike ? 'on' : 'off'}`}>Vol Spike</span>
          <span className={`tag ${s.volatility_breakout ? 'on' : 'off'}`}>BB Break</span>
          <span className={`tag ${isOverbought ? 'down' : isOversold ? 'on' : 'off'}`}>
            {isOverbought ? 'Overbought' : isOversold ? 'Oversold' : 'Neutral RSI'}
          </span>
        </div>

        {/* Confluence meter */}
        <div className="conf-score" style={{ marginTop: 8 }}>
          <ConfluenceMeter score={conf} max={5} />
          <span className="text-muted" style={{ marginLeft: 8 }}>Confluence {conf}/5</span>
        </div>
      </div>
    </div>
  )
}

export default function SignalCards({ signals }: Props) {
  const actionable = signals.filter(s => s.signal !== 'HOLD' && s.probability >= 0.75)
  const all        = signals.slice(0, 12)

  return (
    <div className="card span-3">
      <div className="card-header">
        <span className="card-title"><Zap size={15}/>AI Signals</span>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span className="text-muted">{actionable.length} actionable</span>
          <span className="text-muted" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <BarChart2 size={12}/> {all.length} monitoring
          </span>
        </div>
      </div>
      {all.length === 0
        ? <div className="empty">No signals yet. Train a model first.</div>
        : <div className="signals-grid">{all.map((s,i) => <SignalCard key={i} s={s}/>)}</div>}
    </div>
  )
}
