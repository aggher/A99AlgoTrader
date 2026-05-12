"""
alert_service.py — Sends real-time alerts via Telegram, email, and WebSocket.

Fires alerts when signal probability > threshold AND confluence_score >= 2.
"""
from __future__ import annotations
import asyncio
import logging
import os
import smtplib
from collections import deque
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Deque, List

from dotenv import load_dotenv
load_dotenv()

log                  = logging.getLogger(__name__)
THRESHOLD            = float(os.getenv("SIGNAL_PROBABILITY_THRESHOLD", "0.70"))
_ws_listeners: List[asyncio.Queue] = []
_ws_history:   Deque[dict]         = deque(maxlen=100)


def register_ws_listener(q: asyncio.Queue)   -> None: _ws_listeners.append(q)
def unregister_ws_listener(q: asyncio.Queue) -> None:
    if q in _ws_listeners: _ws_listeners.remove(q)


async def _push_ws(payload: dict) -> None:
    _ws_history.append(payload)
    for q in list(_ws_listeners):
        try: q.put_nowait(payload)
        except asyncio.QueueFull: pass


def _format(signal) -> str:
    d = signal.to_dict() if hasattr(signal, "to_dict") else signal
    sig_label = "[BUY]" if d["signal"] == "BUY" else "[SELL]" if d["signal"] == "SELL" else "[HOLD]"
    return (f"{sig_label} TRADING ALERT\n"
            f"Symbol   : {d['symbol']} [{d.get('timeframe','')}]\n"
            f"Signal   : {d['signal']}\n"
            f"Confidence: {d['probability']:.0%}\n"
            f"Entry    : {d.get('entry')}\n"
            f"Stop Loss: {d.get('stop_loss')}\n"
            f"Take Profit: {d.get('take_profit')}\n"
            f"Confluence: {d.get('confluence_score','-')}/3")


def _telegram(msg: str) -> None:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID",   "")
    if not token or not chat_id: return
    try:
        import httpx
        httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   data={"chat_id": chat_id, "text": msg}, timeout=10)
        log.info("Telegram sent ✓")
    except Exception as e:
        log.error("Telegram error: %s", e)


def _email(subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pw   = os.getenv("SMTP_PASSWORD", "")
    recv = os.getenv("ALERT_RECIPIENT", "")
    if not all([host, user, pw, recv]): return
    try:
        msg            = MIMEMultipart()
        msg["From"]    = user; msg["To"] = recv; msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(host, port) as srv:
            srv.starttls(); srv.login(user, pw)
            srv.sendmail(user, recv, msg.as_string())
        log.info("Email sent ✓")
    except Exception as e:
        log.error("Email error: %s", e)


async def process_signal(signal) -> None:
    d    = signal.to_dict() if hasattr(signal, "to_dict") else signal
    prob = d.get("probability", 0)
    sig  = d.get("signal", "HOLD")
    confl= d.get("confluence_score", 0)

    if sig == "HOLD" or prob < THRESHOLD or confl < 2:
        return

    msg = _format(signal)
    log.info("ALERT: %s", msg)

    await _push_ws({"type": "alert", "data": d})
    _telegram(msg)
    _email(f"[AlgoTrader] {sig} {d['symbol']} {prob:.0%}", msg)


def process_signals_sync(signals) -> None:
    for s in signals:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(process_signal(s))
            else:
                loop.run_until_complete(process_signal(s))
        except Exception:
            # Fallback if no event loop exists in thread
            asyncio.run(process_signal(s))
