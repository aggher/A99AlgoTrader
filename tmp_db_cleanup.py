import sqlite3
import os

db_path = 'data/trading.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    symbols = ('AAPL', 'MSFT', 'TSLA', 'NVDA')
    
    for table in ['ohlcv', 'signals', 'model_metrics']:
        c.execute(f"DELETE FROM {table} WHERE symbol IN (?, ?, ?, ?)", symbols)
        print(f"Purged {table}: {c.rowcount} rows removed.")
        
    conn.commit()
    conn.close()
    print("Database cleanup complete.")
else:
    print("Database not found.")
