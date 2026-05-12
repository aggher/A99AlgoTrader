import { useEffect, useRef, useState } from 'react'
import { createChart, ColorType, CrosshairMode, IChartApi, ISeriesApi } from 'lightweight-charts'
import axios from 'axios'
import { BarChart2 } from 'lucide-react'
import { OHLCVBar } from '../types'

const SYMBOLS = [
  'EURUSD', 'GBPUSD', 'GBPAUD', 'EURAUD', 'GBPNZD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURJPY', 'XAUUSD'
]
const TFS = ['1m', '5m', '15m', '1h', '1d', '1mo']

export default function Chart() {
  const ref     = useRef<HTMLDivElement>(null)
  const chartR  = useRef<IChartApi|null>(null)
  const candleR = useRef<ISeriesApi<'Candlestick'>|null>(null)
  const volR    = useRef<ISeriesApi<'Histogram'>|null>(null)

  const [sym, setSym] = useState('EURUSD')
  const [tf,  setTf]  = useState('1h')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      layout: { background: { type: ColorType.Solid, color: '#0c1322' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#1a2740' }, horzLines: { color: '#1a2740' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1a2740' },
      timeScale: { borderColor: '#1a2740', timeVisible: true, secondsVisible: false },
      width: ref.current.clientWidth,
      height: 340,
    })
    const candle = chart.addCandlestickSeries({
      upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    })
    const vol = chart.addHistogramSeries({
      color: '#388bfd44', priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    vol.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } })

    chartR.current  = chart
    candleR.current = candle
    volR.current    = vol

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth })
    })
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.remove() }
  }, [])

  useEffect(() => {
    setLoading(true)
    const API_URL = import.meta.env.VITE_API_URL || '';
    axios.get(`${API_URL}/api/ohlcv/${sym}?timeframe=${tf}&limit=300`)
      .then(r => {
        const bars: OHLCVBar[] = r.data.bars
        if (!bars?.length) return
        candleR.current?.setData(
          bars.map(b => ({ time: b.time as any, open: b.open, high: b.high, low: b.low, close: b.close }))
        )
        volR.current?.setData(
          bars.map(b => ({
            time: b.time as any, value: b.volume,
            color: b.close >= b.open ? '#26a69a44' : '#ef535044',
          }))
        )
        chartR.current?.timeScale().fitContent()
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [sym, tf])

  return (
    <div className="card span-2">
      <div className="card-header">
        <span className="card-title"><BarChart2 size={15}/>Chart</span>
        <div className="chart-controls">
          <select className="ctrl" value={sym} onChange={e => setSym(e.target.value)}>
            {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          {TFS.map(t => (
            <button key={t} className={`tf-btn${tf === t ? ' active' : ''}`} onClick={() => setTf(t)}>{t}</button>
          ))}
        </div>
      </div>
      <div className="chart-wrap" ref={ref}>
        {loading && <div className="spinner" style={{ position:'absolute', top:'50%', left:'50%', transform:'translate(-50%,-50%)' }}/>}
      </div>
    </div>
  )
}
