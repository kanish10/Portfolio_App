import streamlit as st
import pandas as pd
from datetime import date, timedelta
from src.data_layer import fetch_prices
from src.features   import make_feature_panel
from src.signals    import composite_alpha
from src.portfolio  import weight_long_only, backtest
from src.utils      import performance_stats

# 1) Page config & Dark Theme injection
st.set_page_config(
    page_title="⚡ Portfolio Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Hide Streamlit default menu & footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Adjust main container padding */
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    /* Sidebar padding */
    .css-1lcbmhc {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Dark background for the app */
    .css-18e3th9 { background-color: #0E1117; }  /* main background */
    .css-1v3fvcr { background-color: #0E1117; }  /* sidebar background */

    /* Headers & text color */
    h1, h2, h3, h4, h5 {
        color: #FAFAFA;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    p, span {
        color: #E0E0E0;
    }

    /* Style the ticker selectbox */
    .css-1v3fvcr select {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 8px;
        border: 1px solid #444;
        padding: 0.4rem 0.6rem;
    }
    .css-1v3fvcr option {
        background-color: #262730;
        color: #FAFAFA;
    }

    /* Chart container cards */
    .stLineChart, .stBarChart, .stAltairChart {
        background-color: #1E2127;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Metric cards */
    .stMetric {
        background-color: #1E2127;
        border: 1px solid #3A3F45;
        border-radius: 8px;
        padding: 1rem;
        color: #FAFAFA;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# 2) Sidebar controls
st.sidebar.title("⚙️ Portfolio Coach Settings")
tickers_input = st.sidebar.text_area(
    "Tickers (comma or space)",
    value="NVDA AAPL MSFT C GOOGL",
    help="Enter tickers separated by commas or spaces"
)
tickers = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]

start_dt = st.sidebar.date_input(
    "History from", 
    value=date(2015, 1, 1),
    min_value=date(2000,1,1),
    help="Earliest data pull date"
)
top_cut = st.sidebar.slider(
    "Long-only quantile", 0.05, 0.5, 0.2, 0.05,
    help="What fraction of stocks to go long each month (e.g., 0.2 = top 20%)"
)
# Optional: add a date slider to simulate “train-until” for out-of-sample
cutoff = st.sidebar.date_input(
    "Train until", 
    value=date.today() - timedelta(days=30),
    help="Use only data up to this date to compute signals"
)

# 3) Fetch data (cached via lru_cache in data_layer.py)
prices = fetch_prices(" ".join(tickers), start=start_dt.isoformat())
# Truncate to cutoff date if out-of-sample simulation is desired
if cutoff < date.today():
    prices = prices.loc[:pd.to_datetime(cutoff)]

# 4) Feature & signal pipeline
panel = make_feature_panel(prices)
alpha = composite_alpha(panel)
weights = weight_long_only(alpha, top_cut)
perf = backtest(weights, prices).dropna()

# 5) Main layout
st.markdown("<h1>🚀 Portfolio Coach</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:1.1rem; color:#E0E0E0;'>"
    "Add or remove tickers in the sidebar. The model ranks them by composite alpha "
    "and suggests which to buy (top quantile) and which to sell."
    "</p>",
    unsafe_allow_html=True
)

# 6) Show the “Buy/Sell” signals for the most recent month
latest_date = weights.index[-1]
st.markdown(f"### Model Signals for **{latest_date.date()}**")
latest_w = weights.loc[latest_date]
buy = latest_w[latest_w > 0].index.tolist()
sell = [t for t in tickers if t not in buy]
st.markdown(f"<b style='color:#7FFF00;'>Buy/Add:</b> {buy}", unsafe_allow_html=True)
st.markdown(f"<b style='color:#FF4500;'>Sell/Reduce:</b> {sell}", unsafe_allow_html=True)

# 7) Two-column summary: latest weights & performance stats
col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Weights")
    last_w_df = latest_w.rename("Weight").to_frame()
    # Sort descending
    last_w_df = last_w_df.sort_values("Weight", ascending=False)
    st.bar_chart(last_w_df)

with col2:
    st.subheader("Performance Stats")
    stats = performance_stats(perf)
    st.metric("CAGR", f"{stats.CAGR:.1%}", delta=None)
    st.metric("Vol", f"{stats.Vol:.1%}", delta=None)
    st.metric("Sharpe", f"{stats.Sharpe:.2f}", delta=None)
    st.metric("Max Drawdown", f"{stats.MaxDD:.1%}", delta=None)

# 8) Equity curve (in a dark-themed container)
st.subheader("📈 Equity Curve vs. Benchmark (Buy & Hold)")
cum = (perf + 1).cumprod()
st.line_chart(cum)

# 9) Individual Stock Explorer (more details per ticker)
st.subheader("🔎 Individual Stock Explorer")
sel = st.selectbox(
    label="Choose ticker", 
    options=tickers,
    index=0,
    help="Select a stock to see its price and alpha history"
)
st.line_chart(prices[sel].rename("Price"))

st.markdown(f"<p style='color:#E0E0E0; margin-top:1rem;'>Alpha scores for <b>{sel}</b></p>", unsafe_allow_html=True)
alpha_sel = alpha.xs(sel, level="Ticker").sort_index()
st.line_chart(alpha_sel)

# 10) Footer note
st.markdown(
    "<hr><p style='font-size:0.9rem; color:#999;'>"
    "Built by Kanish Karnad · Powered by Streamlit & Python · "
    "Data from Yahoo Finance · Model = composite alpha (momentum + volatility +…) "
    "</p>",
    unsafe_allow_html=True
)
