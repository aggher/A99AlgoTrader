
import sqlite3
import os

db_path = r"C:\Users\agaton.g\.gemini\antigravity\scratch\trading-platform\data\trading.db"

def check_db():
    if not os.path.exists(db_path):
        print(f"DB not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("--- Latest Signals ---")
    try:
        cursor.execute("SELECT symbol, timeframe, signal, probability, timestamp FROM signals ORDER BY timestamp DESC LIMIT 10;")
        rows = cursor.fetchall()
        if not rows:
            print("No signals found in database.")
        else:
            for row in rows:
                print(row)
    except Exception as e:
        print(f"Error querying signals: {e}")

    print("\n--- Model Metrics (Registered Models) ---")
    try:
        cursor.execute("SELECT symbol, timeframe, accuracy, trained_at FROM model_metrics ORDER BY trained_at DESC LIMIT 5;")
        rows = cursor.fetchall()
        if not rows:
            print("No models registered in model_metrics.")
        else:
            for row in rows:
                print(row)
    except Exception as e:
        print(f"Error querying model_metrics: {e}")

    print("\n--- Latest OHLCV Data ---")
    try:
        cursor.execute("SELECT symbol, timeframe, timestamp, close FROM ohlcv ORDER BY timestamp DESC LIMIT 5;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error querying OHLCV: {e}")

    conn.close()

if __name__ == "__main__":
    check_db()
