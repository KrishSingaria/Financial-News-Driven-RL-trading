from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
import pandas as pd

# === GENERAL CONFIG ===
SINGLE_STOCK = ['AAPL']  
LEXCX_PORTFOLIO = [
    'UNP',   # Union Pacific Corp.
    'BRK-B', # Berkshire Hathaway Inc.
    'MPC',   # Marathon Petroleum Corp.
    'XOM',   # Exxon Mobil Corp.
    'LIN',   # Linde plc
    'PG',    # Procter & Gamble Co.
    'CVX',   # Chevron Corp.
    'CMCSA', # Comcast Corp.
    'PX',    # Praxair Inc.
    'COP',   # ConocoPhillipsDuPont de Nemours Inc
    'GE',    # General Electric Co.
    'HON',   # Honeywell International Inc.
    'IP',    # International Paper Co.
    'JNJ',   # Johnson & Johnson
    'MRK',   # Merck & Co.
    'PFE',   # Pfizer Inc.
    'RTX',   # Raytheon Technologies Corp.
    'TX',    # Texas Industries Inc.
    'WFC',   # Wells Fargo & Co.
    'WY',    # Weyerhaeuser Co.
    'DD'     # DuPont de Nemours Inc.
]

# comment the other one that we dont want
# TICKERS = SINGLE_STOCK
TICKERS = LEXCX_PORTFOLIO
# === DATES ===
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# Parse the dates
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

# Calculate the difference
delta = end_date - start_date

# Get the midpoint by adding half the difference to the start date
middle_date = start_date + delta // 2
# Format the result
middle_date_str = middle_date.strftime("%Y-%m-%d")

TRAIN_START_DATE = START_DATE
TRAIN_END_DATE = middle_date_str
TEST_START_DATE = (middle_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TEST_END_DATE = END_DATE

# === PATHS ===
PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(PARENT_DIR / "data").resolve()
PRICE_DIR = Path(DATA_DIR / "price_data").resolve()
NEWS_DIR = Path(DATA_DIR / "news_data").resolve()
SENTIMENT_DIR = Path(DATA_DIR / "sentiment_data").resolve()
MODEL_DIR = Path(PARENT_DIR / "model").resolve()

# === ENVIRONMENT ===
INITIAL_BALANCE = 10000

# === FINNHUB ===
NEWS_API_URL = "https://finnhub.io/api/v1/company-news"

# === MODEL ===
MODEL_NAME = Path(PARENT_DIR / "models/finbert-tone").resolve()  # Path to the pre-trained model"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
