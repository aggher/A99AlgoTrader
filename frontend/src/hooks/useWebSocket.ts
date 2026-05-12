import { useEffect, useRef, useState } from 'react'
import { SignalData } from '../types'

export function useWebSocket(url: string) {
  const ws        = useRef<WebSocket|null>(null)
  const [connected, setConnected] = useState(false)
  const [alerts,    setAlerts]    = useState<SignalData[]>([])

  useEffect(() => {
    let hb: ReturnType<typeof setInterval>

    function connect() {
      try {
        ws.current = new WebSocket(url)
        ws.current.onopen = () => {
          setConnected(true)
          hb = setInterval(() => {
            if (ws.current?.readyState === WebSocket.OPEN) ws.current.send('ping')
          }, 30_000)
        }
        ws.current.onclose = () => {
          setConnected(false)
          clearInterval(hb)
          setTimeout(connect, 3_000)
        }
        ws.current.onerror = () => ws.current?.close()
        ws.current.onmessage = (e) => {
          try {
            const msg = JSON.parse(e.data)
            if (msg.type === 'alert')
              setAlerts(p => [msg.data as SignalData, ...p].slice(0, 15))
          } catch { /* ignore */ }
        }
      } catch { setTimeout(connect, 5_000) }
    }

    connect()
    return () => { clearInterval(hb); ws.current?.close() }
  }, [url])

  const dismiss = (i: number) => setAlerts(p => p.filter((_,j) => j !== i))
  return { connected, alerts, dismiss }
}
