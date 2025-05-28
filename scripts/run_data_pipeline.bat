@echo off
:: Data pipeline script for financial sentiment analysis
:: This script runs the complete data pipeline:
:: 1. Fetch stock price data
:: 2. Fetch news articles
:: 3. Run sentiment analysis on the news

echo Starting data pipeline...

:: Navigate to the project root directory
cd /d "%~dp0\.."

:: Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo === Step 1: Fetching stock price data ===
python -m scripts.fetch_prices
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to fetch price data
    exit /b 1
)

echo === Step 2: Fetching news articles ===
python -m scripts.fetch_news
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to fetch news data
    exit /b 1
)

echo === Step 3: Running sentiment analysis ===
python -m scripts.sentiment-analysis
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to run sentiment analysis
    exit /b 1
)

echo Data pipeline completed successfully! 