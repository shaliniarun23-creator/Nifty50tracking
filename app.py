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
    page_title="Nifty 50 Strategy Dashboard",
    page_icon="📊",
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
STOP_LOSS_PCT = 0.15


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .dashboard-header {
        padding: 1.7rem 1.8rem;
        border-radius: 18px;
        background: #111827;
        color: #ffffff;
        margin-bottom: 1.3rem;
        border: 1px solid #1f2937;
    }

    .dashboard-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .dashboard-subtitle {
        color: #d1d5db;
        font-size: 0.98rem;
        max-width: 900px;
        line-height: 1.5;
    }

    .strategy-chip {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #f3f4f6;
        color: #111827;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
        margin-top: 1.4rem;
        margin-bottom: 0.2rem;
    }

    .section-caption {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }

    .info-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        color: #374151;
        font-size: 0.92rem;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 14px;
        padding: 1rem;
        color: #92400e;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(17, 24, 39, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        color: #6b7280;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 800;
        color: #111827;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 1rem;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #111827;
        color: white;
    }

    .dataframe {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILS
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
    if pd.isna(value):
        return "NA"
    return f"₹{value:,.0f}"


def format_pct(value):
    if pd.isna(value):
        return "NA"
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
# INDICATORS + 4-RULE STRATEGY
# ============================================================

RULE_COLUMNS = {
    "150 SMA > 220 EMA": "Rule_1_150SMA_GT_220EMA",
    "Close > 50 SMA": "Rule_2_Close_GT_50SMA",
    "50 SMA > 150 SMA": "Rule_3_50SMA_GT_150SMA",
    "Close > 1.25 x 52W Low": "Rule_4_Close_GT_125pct_52WLow",
}


def add_indicators(df):
    df = df.copy()

    df["SMA_50"] = df["Close"].rolling(SMA_50).mean()
    df["SMA_150"] = df["Close"].rolling(SMA_150).mean()
    df["EMA_220"] = df["Close"].ewm(span=EMA_220, adjust=False).mean()

    df["Low_52W"] = df["Low"].rolling(LOOKBACK_52W).min()

    df["Rule_1_150SMA_GT_220EMA"] = df["SMA_150"] > df["EMA_220"]
    df["Rule_2_Close_GT_50SMA"] = df["Close"] > df["SMA_50"]
    df["Rule_3_50SMA_GT_150SMA"] = df["SMA_50"] > df["SMA_150"]
    df["Rule_4_Close_GT_125pct_52WLow"] = df["Close"] > 1.25 * df["Low_52W"]

    df["Rules_Passed"] = df[list(RULE_COLUMNS.values())].sum(axis=1)
    df["Score_%"] = (df["Rules_Passed"] / 4) * 100

    df["Signal_Status"] = np.select(
        [
            df["Rules_Passed"] == 4,
            df["Rules_Passed"] == 3,
        ],
        [
            "BUY CANDIDATE",
            "WATCHLIST",
        ],
        default="AVOID",
    )

    df["Entry_Signal"] = df["Rules_Passed"] == 4

    return df


def build_signal_table(stock_data):
    rows = []

    for symbol, df in stock_data.items():
        if df.empty:
            continue

        last = df.iloc[-1]

        rule_results = {
            rule_name: bool(last[col])
            for rule_name, col in RULE_COLUMNS.items()
        }

        failed_rules = [rule for rule, passed in rule_results.items() if not passed]

        distance_to_50sma = np.nan
        if pd.notna(last["SMA_50"]) and last["SMA_50"] > 0:
            distance_to_50sma = ((last["Close"] / last["SMA_50"]) - 1) * 100

        distance_to_220ema = np.nan
        if pd.notna(last["EMA_220"]) and last["EMA_220"] > 0:
            distance_to_220ema = ((last["Close"] / last["EMA_220"]) - 1) * 100

        pct_above_52w_low = np.nan
        if pd.notna(last["Low_52W"]) and last["Low_52W"] > 0:
            pct_above_52w_low = ((last["Close"] / last["Low_52W"]) - 1) * 100

        row = {
            "Stock": symbol,
            "Status": last["Signal_Status"],
            "Close": round(last["Close"], 2),
            "Rules Passed": int(last["Rules_Passed"]),
            "Score %": round(last["Score_%"], 2),
            "Failed Rules": ", ".join(failed_rules) if failed_rules else "None",
            "% Above 50 SMA": round(distance_to_50sma, 2),
            "% Above 220 EMA": round(distance_to_220ema, 2),
            "% Above 52W Low": round(pct_above_52w_low, 2),
            "SMA 50": round(last["SMA_50"], 2) if pd.notna(last["SMA_50"]) else np.nan,
            "SMA 150": round(last["SMA_150"], 2) if pd.notna(last["SMA_150"]) else np.nan,
            "EMA 220": round(last["EMA_220"], 2) if pd.notna(last["EMA_220"]) else np.nan,
            "52W Low": round(last["Low_52W"], 2) if pd.notna(last["Low_52W"]) else np.nan,
        }

        row.update(rule_results)
        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        status_rank = {"BUY CANDIDATE": 1, "WATCHLIST": 2, "AVOID": 3}
        result["Status Rank"] = result["Status"].map(status_rank)

        result = result.sort_values(
            by=["Status Rank", "Rules Passed", "Score %", "% Above 50 SMA"],
            ascending=[True, False, False, False],
        ).drop(columns=["Status Rank"])

    return result


def calculate_market_breadth(signal_df):
    if signal_df.empty:
        return {
            "Total Stocks": 0,
            "Buy Candidates": 0,
            "Watchlist": 0,
            "Avoid": 0,
            "% Buy Candidates": 0,
            "% Watchlist": 0,
            "% Above 50 SMA": 0,
            "% Above 220 EMA": 0,
        }

    total = len(signal_df)
    buy = len(signal_df[signal_df["Status"] == "BUY CANDIDATE"])
    watch = len(signal_df[signal_df["Status"] == "WATCHLIST"])
    avoid = len(signal_df[signal_df["Status"] == "AVOID"])

    above_50 = signal_df["Close > 50 SMA"].sum()
    above_220 = signal_df["150 SMA > 220 EMA"].sum()

    return {
        "Total Stocks": total,
        "Buy Candidates": buy,
        "Watchlist": watch,
        "Avoid": avoid,
        "% Buy Candidates": round((buy / total) * 100, 2),
        "% Watchlist": round((watch / total) * 100, 2),
        "% Above 50 SMA": round((above_50 / total) * 100, 2),
        "% Above 220 EMA": round((above_220 / total) * 100, 2),
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

        # exits
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

        # entries: enter next day open after 4/4 signal
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

        # portfolio value
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

def status_donut_chart(signal_df):
    status_counts = signal_df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.55,
        title="Signal Classification",
    )

    fig.update_layout(
        height=390,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
    )

    return fig


def rule_pass_chart(signal_df):
    rows = []

    for rule in RULE_COLUMNS.keys():
        rows.append({
            "Rule": rule,
            "Passed Stocks": int(signal_df[rule].sum()),
        })

    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="Rule",
        y="Passed Stocks",
        text="Passed Stocks",
        title="Rule-wise Pass Count",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        height=390,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=80),
        xaxis_tickangle=-20,
    )

    return fig


def score_distribution_chart(signal_df):
    fig = px.histogram(
        signal_df,
        x="Score %",
        nbins=5,
        title="Rule Score Distribution",
    )

    fig.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return fig


def top_score_chart(signal_df):
    top_df = signal_df.sort_values("Score %", ascending=False).head(20)

    fig = px.bar(
        top_df,
        x="Stock",
        y="Score %",
        color="Status",
        text="Score %",
        title="Top Stocks by Rule Score",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        height=460,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=90),
        xaxis_tickangle=-45,
        legend_title_text="",
    )

    return fig


def equity_curve_chart(portfolio_df):
    fig = px.line(
        portfolio_df,
        x="Date",
        y="Portfolio Value",
        title="Portfolio Value Over Time",
    )

    fig.update_traces(line=dict(width=3))

    fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
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
        height=350,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
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

    buy_df = plot_df[plot_df["Entry_Signal"] == True]

    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df["Date"],
            y=buy_df["Close"],
            mode="markers",
            name="4/4 Buy Candidate",
            marker=dict(size=10, symbol="triangle-up"),
        ))

    fig.update_layout(
        title=f"{symbol} Technical Chart",
        height=620,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">Nifty 50 Strategy Dashboard</div>
        <div class="dashboard-subtitle">
            Professional rule-based screening dashboard for Nifty 50 stocks.
            Classifies stocks into Buy Candidate, Watchlist, or Avoid using a 4-rule trend and momentum framework.
        </div>
        <div class="strategy-chip">4-Rule Trend Strength Model</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Settings")

download_start_date = st.sidebar.date_input(
    "Download start date",
    value=pd.to_datetime(DEFAULT_DOWNLOAD_START_DATE).date(),
)

backtest_start_date = st.sidebar.date_input(
    "Backtest start date",
    value=pd.to_datetime(DEFAULT_BACKTEST_START_DATE).date(),
)

watchlist_min_rules = st.sidebar.slider(
    "Watchlist minimum rules passed",
    min_value=1,
    max_value=4,
    value=3,
)

st.sidebar.divider()

st.sidebar.metric("Initial Capital", format_inr(INITIAL_CAPITAL))
st.sidebar.metric("Max Allocation / Stock", format_pct(MAX_POSITION_PCT * 100))
st.sidebar.metric("Stop Loss", format_pct(STOP_LOSS_PCT * 100))

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Strategy Classification**

    - **4/4:** Buy Candidate  
    - **3/4:** Watchlist  
    - **0–2/4:** Avoid  

    **Rules Used**

    1. 150 SMA > 220 EMA  
    2. Close > 50 SMA  
    3. 50 SMA > 150 SMA  
    4. Close > 1.25 × 52W Low  
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

c1, c2, c3 = st.columns([1, 1, 2])

c1.metric("Symbols Loaded", len(symbols))
c2.metric("Universe", "Nifty 50")
c3.markdown(
    f"""
    <div class="info-box">
    Source file detected: <b>{INPUT_FILE}</b>. Keep the CSV in the same folder as <b>app.py</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("View loaded symbols"):
    st.write(symbols)


# ============================================================
# RUN
# ============================================================

run_dashboard = st.button(
    "Download Data and Run Dashboard",
    type="primary",
    use_container_width=True,
)

if not run_dashboard:
    st.markdown(
        """
        <div class="warning-box">
        Click the button above to fetch latest Yahoo Finance data, classify stocks, generate charts, and run the backtest.
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
# BUILD DATA
# ============================================================

signal_df = build_signal_table(stock_data)
breadth = calculate_market_breadth(signal_df)

buy_df = signal_df[signal_df["Status"] == "BUY CANDIDATE"]
watchlist_df = signal_df[signal_df["Rules Passed"] >= watchlist_min_rules]
avoid_df = signal_df[signal_df["Status"] == "AVOID"]

trades_df, portfolio_df = run_backtest(
    stock_data,
    backtest_start_date.strftime("%Y-%m-%d"),
)

if portfolio_df.empty:
    st.error("Backtest could not run because portfolio data is empty.")
    st.stop()

summary = calculate_summary(trades_df, portfolio_df)


# ============================================================
# DASHBOARD KPIs
# ============================================================

st.markdown('<div class="section-title">Market Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Current classification based on the latest available price data.</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Total Stocks", breadth["Total Stocks"])
k2.metric("Buy Candidates", breadth["Buy Candidates"])
k3.metric("Watchlist", breadth["Watchlist"])
k4.metric("Avoid", breadth["Avoid"])
k5.metric("% Above 50 SMA", format_pct(breadth["% Above 50 SMA"]))
k6.metric("% Buy Candidates", format_pct(breadth["% Buy Candidates"]))

st.markdown('<div class="section-title">Backtest Snapshot</div>', unsafe_allow_html=True)

b1, b2, b3, b4, b5, b6 = st.columns(6)

b1.metric("Final Value", format_inr(summary["Final Portfolio Value"]))
b2.metric("Total Return", format_pct(summary["Total Return %"]))
b3.metric("Max Drawdown", format_pct(summary["Max Drawdown %"]))
b4.metric("Trades", summary["Total Trades"])
b5.metric("Win Rate", format_pct(summary["Win Rate %"]))
b6.metric("Profit Factor", summary["Profit Factor"])


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview Charts",
        "Signals",
        "Backtest",
        "Stock Analysis",
        "Downloads",
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown('<div class="section-title">Signal Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(status_donut_chart(signal_df), use_container_width=True)

    with chart_col2:
        st.plotly_chart(rule_pass_chart(signal_df), use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.plotly_chart(score_distribution_chart(signal_df), use_container_width=True)

    with chart_col4:
        st.plotly_chart(top_score_chart(signal_df), use_container_width=True)


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown('<div class="section-title">Buy Candidates</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Stocks passing all 4 strategy rules.</div>',
        unsafe_allow_html=True,
    )

    if buy_df.empty:
        st.warning("No 4/4 buy candidates found today.")
    else:
        st.dataframe(buy_df, use_container_width=True)

    st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Stocks passing the selected minimum number of rules.</div>',
        unsafe_allow_html=True,
    )

    if watchlist_df.empty:
        st.warning("No watchlist stocks found.")
    else:
        st.dataframe(watchlist_df, use_container_width=True)

    st.markdown('<div class="section-title">Full Signal Table</div>', unsafe_allow_html=True)
    st.dataframe(signal_df, use_container_width=True)


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.markdown('<div class="section-title">Backtest Performance</div>', unsafe_allow_html=True)

    st.plotly_chart(equity_curve_chart(portfolio_df), use_container_width=True)
    st.plotly_chart(drawdown_chart(portfolio_df), use_container_width=True)

    st.markdown('<div class="section-title">Trade Log</div>', unsafe_allow_html=True)

    if trades_df.empty:
        st.warning("No historical trades were generated with the current rules.")
    else:
        st.dataframe(trades_df, use_container_width=True)

    st.markdown('<div class="section-title">Portfolio History</div>', unsafe_allow_html=True)
    st.dataframe(portfolio_df, use_container_width=True)


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.markdown('<div class="section-title">Stock-Level Analysis</div>', unsafe_allow_html=True)

    selected_symbol = st.selectbox(
        "Select stock",
        options=list(stock_data.keys()),
    )

    selected_df = stock_data[selected_symbol]
    latest = selected_df.iloc[-1]

    s1, s2, s3, s4, s5 = st.columns(5)

    s1.metric("Latest Close", format_inr(latest["Close"]))
    s2.metric("Status", latest["Signal_Status"])
    s3.metric("Rules Passed", f"{int(latest['Rules_Passed'])}/4")
    s4.metric("Score", format_pct(latest["Score_%"]))
    s5.metric("52W Low", format_inr(latest["Low_52W"]))

    st.plotly_chart(candlestick_chart(selected_df, selected_symbol), use_container_width=True)

    latest_rules = {
        rule: bool(latest[col])
        for rule, col in RULE_COLUMNS.items()
    }

    latest_rules_df = pd.DataFrame(
        [{"Rule": rule, "Passed": passed} for rule, passed in latest_rules.items()]
    )

    st.markdown('<div class="section-title">Latest Rule Check</div>', unsafe_allow_html=True)
    st.dataframe(latest_rules_df, use_container_width=True)


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.markdown('<div class="section-title">Download Outputs</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "Download Buy Candidates",
            data=buy_df.to_csv(index=False),
            file_name="buy_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download Watchlist",
            data=watchlist_df.to_csv(index=False),
            file_name="watchlist.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            "Download Full Signal Table",
            data=signal_df.to_csv(index=False),
            file_name="full_signal_table.csv",
            mime="text/csv",
            use_container_width=True,
        )

    e1, e2 = st.columns(2)

    with e1:
        st.download_button(
            "Download Trade Log",
            data=trades_df.to_csv(index=False),
            file_name="backtest_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e2:
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
        Limitation: this dashboard uses the current Nifty 50 list from your CSV.
        Historical backtest results may have survivorship bias because past index composition changes are not included.
        </div>
        """,
        unsafe_allow_html=True,
    )
