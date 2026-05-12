import { Activity } from 'lucide-react'
import { PriceData } from '../types'

interface Props { prices: PriceData[] }

function fmt(v: number): string {
  if (v >= 100)  return v.toFixed(2)
  if (v >= 1)    return v.toFixed(4)
  return v.toFixed(5)
}

export default function LivePrices({ prices }: Props) {
  return (
    <div className="card span-3">
      <div className="card-header">
        <span className="card-title"><Activity size={15}/>Live Prices</span>
        <span className="text-muted">{prices.length} symbols</span>
      </div>
      {prices.length === 0
        ? <div className="empty">Collecting market data…</div>
        : (
          <div className="prices-grid">
            {prices.map(p => {
              const chg = ((p.close - p.open) / p.open) * 100
              const dir = chg > 0.01 ? 'up' : chg < -0.01 ? 'down' : 'flat'
              return (
                <div className="price-chip" key={p.symbol}>
                  <div className="price-chip-sym">{p.symbol}</div>
                  <div className={`price-chip-val mono ${dir}`}>{fmt(p.close)}</div>
                  <div className={`price-change ${dir}`}>
                    {chg >= 0 ? '+' : ''}{chg.toFixed(3)}%
                  </div>
                </div>
              )
            })}
          </div>
        )
      }
    </div>
  )
}
