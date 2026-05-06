# Nifty 50 Strategy Dashboard

A professional Streamlit dashboard for screening Nifty 50 stocks using a 4-rule technical strategy and backtesting portfolio performance.

## Strategy Rules

A stock qualifies when all 4 rules are satisfied:

1. 150-day SMA > 220-day EMA
2. Close price > 50-day SMA
3. 50-day SMA > 150-day SMA
4. Close price > 1.25 × 52-week low

## Features

- Downloads Nifty 50 stock data from Yahoo Finance
- Calculates SMA, EMA, 52-week low, and rule scores
- Shows exact qualified stocks
- Shows near-qualified watchlist
- Provides market breadth analytics
- Runs portfolio backtest
- Displays equity curve and drawdown chart
- Provides stock-level technical chart
- Allows CSV download of outputs

## Files Required

```text
app.py
requirements.txt
README.md
Nifty 50 symbols.csv
