import { SignalData } from '../types'

interface Props {
  alerts:  SignalData[]
  dismiss: (i: number) => void
}

function fmt(v: number|null|undefined): string {
  if (v == null) return '—'
  if (v >= 100)  return v.toFixed(2)
  return v.toFixed(5)
}

export default function AlertPanel({ alerts, dismiss }: Props) {
  if (alerts.length === 0) return null
  return (
    <div className="toasts">
      {alerts.map((a, i) => (
        <div key={i} className={`toast ${a.signal.toLowerCase()}`}>
          <div className="toast-head">
            <strong style={{ fontSize:12 }}>
              {a.signal === 'BUY' ? '🟢' : '🔴'} {a.signal} {a.symbol}
            </strong>
            <button className="toast-close" onClick={() => dismiss(i)}>×</button>
          </div>
          <div className="text-muted">Confidence: {Math.round(a.probability*100)}%</div>
          <div className="text-muted">Entry: {fmt(a.entry)} | TP: {fmt(a.take_profit)}</div>
        </div>
      ))}
    </div>
  )
}
