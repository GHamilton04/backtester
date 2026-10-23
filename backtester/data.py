from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_prices(ticker: str, start:str, end:str) -> pd.DataFrame:
    """
    Load daily historical OHLCV data for a given ticker from Yahoo Finance.
    
    Uses a saved CSV file if it exists, otherwise downloads the data and saves
    it to a CSV file for later use.
    """
    file_path = DATA_DIR / f"{ticker}_{start}_{end}.csv"

    if file_path.exists():
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        print(f"Loaded {ticker} from cache: {file_path.name}")
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, multi_level_index=False)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(file_path)

    return df