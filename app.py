import os
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nifty 50 Strategy Command Center",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Nifty 50 symbols.csv"
OUTPUT_FOLDER = "data"

INITIAL_CAPITAL = 1_000_000
MAX_POSITION_PCT = 0.10

DEFAULT_DOWNLOAD_START_DATE = "2023-01-01"
DEFAULT_BACKTEST_START_DATE = "2025-01-01"

SMA_50 = 50
SMA_150 = 150
EMA_220 = 220
LOOKBACK_52W = 252
DIP_LOOKBACK = 90
STOP_LOSS_PCT = 0.15


# ============================================================
# CSS DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        max-width: 850px;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e2e8f0;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 1.8rem;
        margin-bottom: 0.6rem;
    }

    .section-caption {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .info-card {
        padding: 1rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .note-box {
        padding: 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.95rem;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.88rem;
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.45rem 1rem;
        background-color: #f1f5f9;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0f172a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))


def to_yahoo_symbol(symbol):
    symbol = str(symbol).strip().upper()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def read_symbols(file_path):
    df = pd.read_csv(file_path, header=None)
    symbols = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

    if not symbols:
        raise ValueError("No symbols found in CSV.")

    return symbols


def format_inr(value):
    return f"₹{value:,.0f}"


def format_pct(value):
    return f"{value:.2f}%"


@st.cache_data(show_spinner=False)
def download_stock_data(ticker, start_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=date.today().strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ============================================================
# INDICATORS + STRATEGY
# ============================================================

def add_indicators(df):
    df = df.copy()

    df["SMA_50"] = df["Close"].rolling(SMA_50).mean()
    df["SMA_150"] = df["Close"].rolling(SMA_150).mean()
    df["EMA_220"] = df["Close"].ewm(span=EMA_220, adjust=False).mean()

    df["Low_52W"] = df["Low"].rolling(LOOKBACK_52W).min()
    df["High_52W_Prev"] = df["Close"].rolling(LOOKBACK_52W).max().shift(1)

    df["Dipped_Below_EMA_220"] = (
        (df["Low"] < df["EMA_220"])
        .rolling(DIP_LOOKBACK)
        .max()
        .fillna(0)
        .astype(bool)
    )

    df["Rule_1_150SMA_GT_220EMA"] = df["SMA_150"] > df["EMA_220"]
    df["Rule_2_Close_GT_50SMA"] = df["Close"] > df["SMA_50"]
    df["Rule_3_50SMA_GT_150SMA"] = df["SMA_50"] > df["SMA_150"]
    df["Rule_4_Close_GT_125pct_52WLow"] = df["Close"] > 1.25 * df["Low_52W"]
    df["Rule_5_Dipped_Below_220EMA_90D"] = df["Dipped_Below_EMA_220"]
    df["Rule_6_Close_Breakout_52WHigh"] = df["Close"] > df["High_52W_Prev"]

    df["Filter_Pass"] = (
        df["Rule_1_150SMA_GT_220EMA"]
        & df["Rule_2_Close_GT_50SMA"]
        & df["Rule_3_50SMA_GT_150SMA"]
        & df["Rule_4_Close_GT_125pct_52WLow"]
        & df["Rule_5_Dipped_Below_220EMA_90D"]
    )

    df["Entry_Signal"] = df["Filter_Pass"] & df["Rule_6_Close_Breakout_52WHigh"]

    return df


RULE_COLUMNS = {
    "150 SMA > 220 EMA": "Rule_1_150SMA_GT_220EMA",
    "Close > 50 SMA": "Rule_2_Close_GT_50SMA",
    "50 SMA > 150 SMA": "Rule_3_50SMA_GT_150SMA",
    "Close > 1.25 x 52W Low": "Rule_4_Close_GT_125pct_52WLow",
    "Dipped below 220 EMA in 90D": "Rule_5_Dipped_Below_220EMA_90D",
    "Close breakout above 52W High": "Rule_6_Close_Breakout_52WHigh",
}


def get_today_buy_candidates(stock_data):
    rows = []

    for symbol, df in stock_data.items():
        if df.empty:
            continue

        last = df.iloc[-1]

        if bool(last["Entry_Signal"]):
            rows.append({
                "Stock": symbol,
                "Close": round(last["Close"], 2),
                "SMA 50": round(last["SMA_50"], 2),
                "SMA 150": round(last["SMA_150"], 2),
                "EMA 220": round(last["EMA_220"], 2),
                "52W High Prev": round(last["High_52W_Prev"], 2),
                "52W Low": round(last["Low_52W"], 2),
                "% Above 52W Low": round(((last["Close"] / last["Low_52W"]) - 1) * 100, 2),
            })

    return pd.DataFrame(rows)


def get_near_trade_watchlist(stock_data):
    rows = []

    for symbol, df in stock_data.items():
        if df.empty:
            continue

        last = df.iloc[-1]

        rule_results = {
            rule_name: bool(last[col])
            for rule_name, col in RULE_COLUMNS.items()
        }

        passed_count = sum(rule_results.values())
        failed_rules = [rule for rule, passed in rule_results.items() if not passed]

        distance_to_breakout = np.nan
        if pd.notna(last["High_52W_Prev"]) and last["High_52W_Prev"] > 0:
            distance_to_breakout = ((last["Close"] / last["High_52W_Prev"]) - 1) * 100

        distance_to_220ema = np.nan
        if pd.notna(last["EMA_220"]) and last["EMA_220"] > 0:
            distance_to_220ema = ((last["Close"] / last["EMA_220"]) - 1) * 100

        row = {
            "Stock": symbol,
            "Close": round(last["Close"], 2),
            "Rules Passed": passed_count,
            "Total Rules": 6,
            "Score %": round((passed_count / 6) * 100, 2),
            "Failed Rules": ", ".join(failed_rules) if failed_rules else "None",
            "% From 52W High Breakout": round(distance_to_breakout, 2),
            "% Above 220 EMA": round(distance_to_220ema, 2),
            "Exact Buy Signal": bool(last["Entry_Signal"]),
        }

        row.update(rule_results)
        rows.append(row)

    watchlist = pd.DataFrame(rows)

    if not watchlist.empty:
        watchlist = watchlist.sort_values(
            by=["Exact Buy Signal", "Rules Passed", "% From 52W High Breakout"],
            ascending=[False, False, False],
        )

    return watchlist


def calculate_market_breadth(stock_data):
    total = 0
    above_50 = 0
    above_220 = 0
    exact_signals = 0
    near_trades = 0

    for _, df in stock_data.items():
        if df.empty:
            continue

        last = df.iloc[-1]
        total += 1

        if pd.notna(last["SMA_50"]) and last["Close"] > last["SMA_50"]:
            above_50 += 1

        if pd.notna(last["EMA_220"]) and last["Close"] > last["EMA_220"]:
            above_220 += 1

        if bool(last["Entry_Signal"]):
            exact_signals += 1

        rules_passed = sum(bool(last[col]) for col in RULE_COLUMNS.values())
        if rules_passed >= 4:
            near_trades += 1

    if total == 0:
        return {
            "Total Stocks": 0,
            "% Above 50 SMA": 0,
            "% Above 220 EMA": 0,
            "Exact Signals": 0,
            "Near Trades": 0,
        }

    return {
        "Total Stocks": total,
        "% Above 50 SMA": round((above_50 / total) * 100, 2),
        "% Above 220 EMA": round((above_220 / total) * 100, 2),
        "Exact Signals": exact_signals,
        "Near Trades": near_trades,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(stock_data, backtest_start_date):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    portfolio_values = []

    all_dates = sorted(set(d for df in stock_data.values() for d in df["Date"]))
    all_dates = [d for d in all_dates if d >= pd.to_datetime(backtest_start_date)]

    for current_date in all_dates:

        # EXIT LOGIC
        for symbol in list(positions.keys()):
            df = stock_data[symbol]
            row = df[df["Date"] == current_date]

            if row.empty:
                continue

            row = row.iloc[0]
            position = positions[symbol]

            exit_reason = None

            if row["Close"] < row["EMA_220"]:
                exit_reason = "Close below 220 EMA"
            elif row["Close"] <= position["Entry_Price"] * (1 - STOP_LOSS_PCT):
                exit_reason = "15% stop loss"

            if exit_reason:
                exit_price = row["Close"]
                proceeds = position["Shares"] * exit_price
                cash += proceeds

                pnl = proceeds - position["Invested"]
                pnl_pct = pnl / position["Invested"]

                trades.append({
                    "Symbol": symbol,
                    "Entry Date": position["Entry_Date"],
                    "Entry Price": round(position["Entry_Price"], 2),
                    "Exit Date": current_date,
                    "Exit Price": round(exit_price, 2),
                    "Shares": position["Shares"],
                    "Invested": round(position["Invested"], 2),
                    "PnL": round(pnl, 2),
                    "PnL %": round(pnl_pct * 100, 2),
                    "Exit Reason": exit_reason,
                })

                del positions[symbol]

        # ENTRY LOGIC
        max_allocation = INITIAL_CAPITAL * MAX_POSITION_PCT

        for symbol, df in stock_data.items():
            if symbol in positions:
                continue

            signal_row = df[df["Date"] == current_date]

            if signal_row.empty:
                continue

            signal_row = signal_row.iloc[0]

            if not bool(signal_row["Entry_Signal"]):
                continue

            future_rows = df[df["Date"] > current_date]

            if future_rows.empty:
                continue

            entry_row = future_rows.iloc[0]
            entry_date = entry_row["Date"]
            entry_price = entry_row["Open"]

            if pd.isna(entry_price) or entry_price <= 0:
                continue

            allocation = min(max_allocation, cash)
            shares = int(allocation // entry_price)

            if shares <= 0:
                continue

            invested = shares * entry_price
            cash -= invested

            positions[symbol] = {
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
                "Shares": shares,
                "Invested": invested,
            }

        # PORTFOLIO VALUE
        portfolio_value = cash

        for symbol, position in positions.items():
            df = stock_data[symbol]
            row = df[df["Date"] == current_date]

            if not row.empty:
                portfolio_value += position["Shares"] * row.iloc[0]["Close"]

        portfolio_values.append({
            "Date": current_date,
            "Portfolio Value": portfolio_value,
            "Cash": cash,
            "Open Positions": len(positions),
        })

    return pd.DataFrame(trades), pd.DataFrame(portfolio_values)


def calculate_summary(trades_df, portfolio_df):
    if portfolio_df.empty:
        return {}

    final_value = portfolio_df["Portfolio Value"].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    portfolio_df["Daily Return"] = portfolio_df["Portfolio Value"].pct_change()
    portfolio_df["Peak"] = portfolio_df["Portfolio Value"].cummax()
    portfolio_df["Drawdown"] = (
        portfolio_df["Portfolio Value"] - portfolio_df["Peak"]
    ) / portfolio_df["Peak"]

    max_drawdown = portfolio_df["Drawdown"].min()

    if trades_df.empty:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    else:
        wins = trades_df[trades_df["PnL"] > 0]
        losses = trades_df[trades_df["PnL"] <= 0]

        win_rate = len(wins) / len(trades_df)
        avg_win = wins["PnL %"].mean() if not wins.empty else 0
        avg_loss = losses["PnL %"].mean() if not losses.empty else 0

        gross_profit = wins["PnL"].sum() if not wins.empty else 0
        gross_loss = abs(losses["PnL"].sum()) if not losses.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "Initial Capital": INITIAL_CAPITAL,
        "Final Portfolio Value": round(final_value, 2),
        "Total Return %": round(total_return * 100, 2),
        "Max Drawdown %": round(max_drawdown * 100, 2),
        "Total Trades": len(trades_df),
        "Win Rate %": round(win_rate * 100, 2),
        "Average Win %": round(avg_win, 2),
        "Average Loss %": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
    }


# ============================================================
# CHARTS
# ============================================================

def equity_curve_chart(portfolio_df):
    fig = px.line(
        portfolio_df,
        x="Date",
        y="Portfolio Value",
        title="Portfolio Value Over Time",
    )

    fig.update_traces(line=dict(width=3))

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        title_font=dict(size=20),
    )

    return fig


def drawdown_chart(portfolio_df):
    fig = px.area(
        portfolio_df,
        x="Date",
        y="Drawdown",
        title="Portfolio Drawdown",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        title_font=dict(size=20),
    )

    return fig


def rule_score_chart(watchlist_df):
    if watchlist_df.empty:
        return None

    top_df = watchlist_df.sort_values("Score %", ascending=False).head(15)

    fig = px.bar(
        top_df,
        x="Stock",
        y="Score %",
        title="Top Watchlist Stocks by Rule Score",
        text="Score %",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=55, b=80),
        template="plotly_white",
        xaxis_tickangle=-45,
        title_font=dict(size=20),
    )

    return fig


def candlestick_chart(df, symbol):
    plot_df = df.tail(280).copy()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=plot_df["Date"],
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="Price",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["Date"],
        y=plot_df["SMA_50"],
        mode="lines",
        name="SMA 50",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["Date"],
        y=plot_df["SMA_150"],
        mode="lines",
        name="SMA 150",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["Date"],
        y=plot_df["EMA_220"],
        mode="lines",
        name="EMA 220",
    ))

    signal_df = plot_df[plot_df["Entry_Signal"] == True]

    if not signal_df.empty:
        fig.add_trace(go.Scatter(
            x=signal_df["Date"],
            y=signal_df["Close"],
            mode="markers",
            name="Buy Signal",
            marker=dict(size=11, symbol="triangle-up"),
        ))

    fig.update_layout(
        title=f"{symbol} Price Chart with Strategy Signals",
        height=620,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
        title_font=dict(size=20),
    )

    return fig


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">NIFTY 50 STRATEGY DASHBOARD</div>
        <div class="hero-title">Nifty 50 Strategy Command Center</div>
        <div class="hero-subtitle">
            A rule-based momentum and breakout dashboard for Indian equities.
            Tracks exact buy signals, near-trade candidates, market breadth, portfolio backtest performance,
            drawdowns, and trade logs using Yahoo Finance data.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Control Panel")

download_start_date = st.sidebar.date_input(
    "Download start date",
    value=pd.to_datetime(DEFAULT_DOWNLOAD_START_DATE).date(),
)

backtest_start_date = st.sidebar.date_input(
    "Backtest start date",
    value=pd.to_datetime(DEFAULT_BACKTEST_START_DATE).date(),
)

min_rules_for_watchlist = st.sidebar.slider(
    "Minimum rules passed for near-trade watchlist",
    min_value=1,
    max_value=6,
    value=4,
)

st.sidebar.divider()

st.sidebar.metric("Initial Capital", format_inr(INITIAL_CAPITAL))
st.sidebar.metric("Max Allocation / Stock", format_pct(MAX_POSITION_PCT * 100))
st.sidebar.metric("Stop Loss", format_pct(STOP_LOSS_PCT * 100))

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Strategy Rules**
    1. 150 SMA > 220 EMA  
    2. Close > 50 SMA  
    3. 50 SMA > 150 SMA  
    4. Close > 1.25 × 52W low  
    5. Low dipped below 220 EMA in 90 days  
    6. Close breaks above previous 52W high  
    """
)


# ============================================================
# LOAD SYMBOLS
# ============================================================

if not os.path.exists(INPUT_FILE):
    st.error(f"CSV file not found: `{INPUT_FILE}`")
    st.stop()

try:
    symbols = read_symbols(INPUT_FILE)
except Exception as e:
    st.error(f"Error reading symbols: {e}")
    st.stop()

top_col1, top_col2, top_col3 = st.columns([1.2, 1.2, 2])

with top_col1:
    st.metric("Symbols Loaded", len(symbols))

with top_col2:
    st.metric("Data Source", "Yahoo Finance")

with top_col3:
    st.markdown(
        f"""
        <div class="note-box">
        Loaded symbols from <b>{INPUT_FILE}</b>. Keep this CSV in the same folder as app.py.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("View loaded symbols"):
    st.write(symbols)


# ============================================================
# RUN DASHBOARD
# ============================================================

run_button = st.button("Download Data and Run Dashboard", type="primary", use_container_width=True)

if not run_button:
    st.markdown(
        """
        <div class="warning-box">
        Click <b>Download Data and Run Dashboard</b> to fetch latest data, generate signals, and run the backtest.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


stock_data = {}
failed_downloads = []

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

progress = st.progress(0)
status = st.empty()

for i, symbol in enumerate(symbols):
    ticker = to_yahoo_symbol(symbol)
    status.write(f"Downloading {ticker}...")

    try:
        df = download_stock_data(ticker, download_start_date.strftime("%Y-%m-%d"))

        if df.empty:
            failed_downloads.append([symbol, ticker, "No data returned"])
        else:
            df = add_indicators(df)
            stock_data[ticker] = df

            df.to_csv(
                os.path.join(OUTPUT_FOLDER, clean_filename(ticker) + ".csv"),
                index=False,
            )

    except Exception as e:
        failed_downloads.append([symbol, ticker, str(e)])

    progress.progress((i + 1) / len(symbols))

status.write("Download completed.")

if failed_downloads:
    st.warning("Some symbols failed to download.")
    st.dataframe(
        pd.DataFrame(failed_downloads, columns=["Symbol", "Yahoo Ticker", "Reason"]),
        use_container_width=True,
    )

if not stock_data:
    st.error("No stock data available.")
    st.stop()


# ============================================================
# COMPUTE OUTPUTS
# ============================================================

candidates_df = get_today_buy_candidates(stock_data)
watchlist_df = get_near_trade_watchlist(stock_data)
near_df = watchlist_df[watchlist_df["Rules Passed"] >= min_rules_for_watchlist]

breadth = calculate_market_breadth(stock_data)

trades_df, portfolio_df = run_backtest(
    stock_data,
    backtest_start_date.strftime("%Y-%m-%d"),
)

if portfolio_df.empty:
    st.error("Backtest could not run because portfolio data is empty.")
    st.stop()

summary = calculate_summary(trades_df, portfolio_df)


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Live market screening summary and portfolio backtest snapshot.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Total Stocks", breadth["Total Stocks"])
m2.metric("Exact Signals", breadth["Exact Signals"])
m3.metric("Near Trades", breadth["Near Trades"])
m4.metric("% Above 50 SMA", format_pct(breadth["% Above 50 SMA"]))
m5.metric("% Above 220 EMA", format_pct(breadth["% Above 220 EMA"]))

b1, b2, b3, b4, b5 = st.columns(5)

b1.metric("Final Value", format_inr(summary["Final Portfolio Value"]))
b2.metric("Total Return", format_pct(summary["Total Return %"]))
b3.metric("Max Drawdown", format_pct(summary["Max Drawdown %"]))
b4.metric("Total Trades", summary["Total Trades"])
b5.metric("Profit Factor", summary["Profit Factor"])


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Market Signals",
        "Watchlist",
        "Backtest",
        "Stock Chart",
        "Downloads",
    ]
)


# ============================================================
# TAB 1: MARKET SIGNALS
# ============================================================

with tab1:
    st.markdown('<div class="section-title">Today’s Exact Buy Candidates</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Stocks that meet all strategy rules today.</div>',
        unsafe_allow_html=True,
    )

    if candidates_df.empty:
        st.warning("No stocks meet all strategy criteria today.")
    else:
        st.success(f"{len(candidates_df)} stocks qualify today.")
        st.dataframe(candidates_df, use_container_width=True)

    st.markdown('<div class="section-title">Market Breadth Summary</div>', unsafe_allow_html=True)

    breadth_df = pd.DataFrame([breadth])
    st.dataframe(breadth_df, use_container_width=True)


# ============================================================
# TAB 2: WATCHLIST
# ============================================================

with tab2:
    st.markdown('<div class="section-title">Near-Trade Watchlist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Stocks passing most rules but not necessarily all rules. Use this as a monitoring list, not a buy list.</div>',
        unsafe_allow_html=True,
    )

    if near_df.empty:
        st.warning("No stocks qualify for the near-trade watchlist.")
    else:
        st.dataframe(near_df, use_container_width=True)

    fig = rule_score_chart(watchlist_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3: BACKTEST
# ============================================================

with tab3:
    st.markdown('<div class="section-title">Backtest Performance</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Final Portfolio Value", format_inr(summary["Final Portfolio Value"]))
    p2.metric("Total Return", format_pct(summary["Total Return %"]))
    p3.metric("Max Drawdown", format_pct(summary["Max Drawdown %"]))
    p4.metric("Win Rate", format_pct(summary["Win Rate %"]))

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("Total Trades", summary["Total Trades"])
    p6.metric("Average Win", format_pct(summary["Average Win %"]))
    p7.metric("Average Loss", format_pct(summary["Average Loss %"]))
    p8.metric("Profit Factor", summary["Profit Factor"])

    st.plotly_chart(equity_curve_chart(portfolio_df), use_container_width=True)
    st.plotly_chart(drawdown_chart(portfolio_df), use_container_width=True)

    st.markdown('<div class="section-title">Trade Log</div>', unsafe_allow_html=True)

    if trades_df.empty:
        st.warning("No historical trades were generated with the current rules.")
    else:
        st.dataframe(trades_df, use_container_width=True)

    st.markdown('<div class="section-title">Portfolio Data</div>', unsafe_allow_html=True)
    st.dataframe(portfolio_df, use_container_width=True)


# ============================================================
# TAB 4: STOCK CHART
# ============================================================

with tab4:
    st.markdown('<div class="section-title">Stock-Level Technical Chart</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Select a stock to view candlestick movement, SMA 50, SMA 150, EMA 220, and buy signal markers.</div>',
        unsafe_allow_html=True,
    )

    selected_symbol = st.selectbox(
        "Select stock",
        options=list(stock_data.keys()),
    )

    selected_df = stock_data[selected_symbol]

    c1, c2, c3, c4 = st.columns(4)
    latest = selected_df.iloc[-1]

    c1.metric("Latest Close", format_inr(latest["Close"]))
    c2.metric("SMA 50", format_inr(latest["SMA_50"]) if pd.notna(latest["SMA_50"]) else "NA")
    c3.metric("SMA 150", format_inr(latest["SMA_150"]) if pd.notna(latest["SMA_150"]) else "NA")
    c4.metric("EMA 220", format_inr(latest["EMA_220"]) if pd.notna(latest["EMA_220"]) else "NA")

    st.plotly_chart(candlestick_chart(selected_df, selected_symbol), use_container_width=True)

    st.markdown('<div class="section-title">Latest Rule Check</div>', unsafe_allow_html=True)

    latest_rules = {
        rule: bool(latest[col])
        for rule, col in RULE_COLUMNS.items()
    }

    latest_rules_df = pd.DataFrame(
        [{"Rule": rule, "Passed": passed} for rule, passed in latest_rules.items()]
    )

    st.dataframe(latest_rules_df, use_container_width=True)


# ============================================================
# TAB 5: DOWNLOADS
# ============================================================

with tab5:
    st.markdown('<div class="section-title">Download Outputs</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "Download Exact Buy Candidates",
            data=candidates_df.to_csv(index=False),
            file_name="today_buy_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download Near-Trade Watchlist",
            data=near_df.to_csv(index=False),
            file_name="near_trade_watchlist.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            "Download Trade Log",
            data=trades_df.to_csv(index=False),
            file_name="backtest_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.download_button(
        "Download Portfolio History",
        data=portfolio_df.to_csv(index=False),
        file_name="backtest_portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="warning-box">
        Important limitation: this dashboard uses current Nifty 50 symbols from your CSV.
        Historical backtest results may have survivorship bias because past Nifty 50 index composition changes are not included.
        </div>
        """,
        unsafe_allow_html=True,
    )
