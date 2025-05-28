import config
import yfinance as yf
import pandas as pd
import time
config.PRICE_DIR.mkdir(parents=True, exist_ok=True)

def fixDF(df_path, ticker):
    df = pd.read_csv(df_path)

    # Drop row if it's a repeated header like "AAPL,AAPL,..."
    first_row = df.iloc[0].astype(str)
    if all(first_row.get(col, '') == ticker for col in ['Open', 'High', 'Low', 'Close']):
        print(f"Dropping stray header row for {ticker}")
        df = df.drop(index=0).reset_index(drop=True)

    # Keep only expected columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.to_csv(df_path, index=False)



for ticker in config.TICKERS:
    print(f"Fetching: {ticker}")
    
    out_path = config.PRICE_DIR / f"{ticker}.csv"
    if out_path.exists():
        out_path.unlink()

    df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, group_by='column',auto_adjust=True)
    df.reset_index(inplace=True)  # Converts index to 'Date' column
    df.to_csv(out_path, index=False)  # Save with 'Date' as a proper column
    fixDF(out_path, ticker)
    print(f"Saved: {out_path}")
    time.sleep(2)
