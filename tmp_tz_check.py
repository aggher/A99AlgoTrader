import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def check_tz(symbol="GBPUSD", interval="1m", period="1d"):
    print(f"Checking {symbol} {interval} {period}...")
    
    # Primary
    df_p = yf.download(symbol + "=X", period=period, interval=interval, progress=False)
    print(f"Primary Index TZ: {df_p.index.tz}")
    print(f"Primary Head:\n{df_p.index[:3]}")
    
    # Secondary
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    tw_symbol = f"{symbol[:3]}/{symbol[3:]}"
    url = f"https://api.twelvedata.com/time_series?symbol={tw_symbol}&interval=1min&outputsize=10&apikey={api_key}"
    resp = requests.get(url).json()
    if resp.get("status") == "error":
        print(f"Twelve Error: {resp.get('message')}")
        return
        
    df_s = pd.DataFrame(resp.get("values", []))
    df_s['datetime'] = pd.to_datetime(df_s['datetime'])
    print(f"Twelve Data Raw Head:\n{df_s['datetime'].head(3)}")
    
    # Meta
    meta_url = f"https://api.twelvedata.com/symbol_search?symbol={tw_symbol}&apikey={api_key}"
    meta = requests.get(meta_url).json()
    print(f"Twelve Data Meta:\n{meta.get('data', [])[:1]}")

if __name__ == "__main__":
    check_tz()
