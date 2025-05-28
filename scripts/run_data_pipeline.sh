#!/bin/bash

# Data pipeline script for financial sentiment analysis
# This script runs the complete data pipeline:
# 1. Fetch stock price data
# 2. Fetch news articles
# 3. Run sentiment analysis on the news

echo "Starting data pipeline..."

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "=== Step 1: Fetching stock price data ==="
python -m scripts.fetch_prices
if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch price data"
    exit 1
fi

echo "=== Step 2: Fetching news articles ==="
python -m scripts.fetch_news
if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch news data"
    exit 1
fi

echo "=== Step 3: Running sentiment analysis ==="
python -m scripts.sentiment-analysis
if [ $? -ne 0 ]; then
    echo "Error: Failed to run sentiment analysis"
    exit 1
fi

echo "Data pipeline completed successfully!" 