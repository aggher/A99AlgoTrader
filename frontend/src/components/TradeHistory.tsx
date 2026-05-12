import { History } from 'lucide-react'
import { SignalData } from '../types'

interface Props { signals: SignalData[] }

function fmtTime(ts: string): string {
  try { return new Date(ts).toLocaleString(undefined, { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }) }
  catch { return ts }
}

function fmt(v: number|null|undefined): string {
  if (v == null) return '—'
  if (v >= 100) return v.toFixed(2)
  return v.toFixed(5)
}

export default function TradeHistory({ signals }: Props) {
  const rows = signals.filter(s => s.signal !== 'HOLD').slice(0, 50)
  return (
    <div className="card span-2">
      <div className="card-header">
        <span className="card-title"><History size={15}/>Signal History</span>
        <span className="text-muted">{rows.length} signals</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Symbol</th><th>TF</th><th>Signal</th>
              <th>Prob</th><th>Entry</th><th>SL</th><th>TP</th><th>Size</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0
              ? <tr><td colSpan={9} className="empty">No signals yet.</td></tr>
              : rows.map((s, i) => (
                <tr key={i}>
                  <td className="text-muted">{fmtTime(s.timestamp)}</td>
                  <td className="td-mono">{s.symbol}</td>
                  <td className="text-muted">{s.timeframe}</td>
                  <td className={`td-${s.signal.toLowerCase()}`}>{s.signal}</td>
                  <td className="td-mono">{Math.round(s.probability*100)}%</td>
                  <td className="td-mono">{fmt(s.entry)}</td>
                  <td className="td-neg">{fmt(s.stop_loss)}</td>
                  <td className="td-pos">{fmt(s.take_profit)}</td>
                  <td className="text-muted">{s.position_size?.toFixed(2) ?? '—'}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
