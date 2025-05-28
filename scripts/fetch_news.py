import config

import requests
import json
from datetime import datetime, timedelta

# === CONFIG ===
API_KEY = config.FINNHUB_API_KEY
if API_KEY is None:
    raise ValueError("API key not found! Set FINNHUB_API_KEY in .env")
TICKERS = config.TICKERS
config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
def daterange(start_date, end_date):
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

def fetch_news(ticker, date):
    url = config.NEWS_API_URL
    params = {
        'symbol': ticker,
        'from': date.strftime('%Y-%m-%d'),
        'to': date.strftime('%Y-%m-%d'),
        'token': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"[ERROR] {ticker} on {date} => {response.status_code}")
        return []



def main():
    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end = datetime.strptime(config.END_DATE, "%Y-%m-%d")

    for ticker in TICKERS:
        ticker_dir = config.NEWS_DIR / ticker
        ticker_dir.mkdir(exist_ok=True)
        print(f"Fetching news for {ticker}...")

        for single_date in daterange(start, end):
            filename = ticker_dir / f"{single_date.strftime('%Y-%m-%d')}.json"
            if filename.exists():
                continue  # Skip already downloaded
            try:
                news = fetch_news(ticker, single_date)
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(news, f, indent=2)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"[FAIL] {ticker} on {single_date}: {e}")

if __name__ == "__main__":
    main()
