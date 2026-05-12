from backend.train_xgboost_model import train
from backend.data_collector import ALL_SYMBOLS

if __name__ == "__main__":
    # Resuming from the symbol after GBPCHF
    start_symbol = "GBPAUD"
    try:
        idx = ALL_SYMBOLS.index(start_symbol)
        remaining = ALL_SYMBOLS[idx:]
        print(f"Resuming training for: {remaining}")
        train(symbols=remaining, tune=True)
    except ValueError:
        print(f"Symbol {start_symbol} not found in ALL_SYMBOLS. Training all.")
        train(tune=True)
