import streamlit as st, pandas as pd
from datetime import date
from src.data_layer import fetch_prices
from src.features   import make_feature_panel
from src.signals    import composite_alpha
from src.portfolio  import weight_long_only, backtest
from src.utils      import performance_stats

st.set_page_config(page_title="Portfolio Coach", layout="wide")

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Settings")

tickers_input = st.sidebar.text_area(
    "Tickers (comma or space)",
    value="NVDA AAPL MSFT C GOOGL"
)
tickers = [t.strip().upper() for t in tickers_input.replace(",", " ").split()]
start_dt = st.sidebar.date_input("History from", date(2015, 1, 1))
top_cut  = st.sidebar.slider("Top quantile long-only", 0.05, 0.5, 0.2, 0.05)

# ---- DATA ----
prices = fetch_prices(" ".join(tickers), start=start_dt.isoformat())
st.subheader("Price preview")
st.dataframe(prices.tail())

panel  = make_feature_panel(prices)
alpha  = composite_alpha(panel)
weights = weight_long_only(alpha, top_cut)
perf    = backtest(weights, prices).dropna()

# ---- DASHBOARD ----
col1, col2 = st.columns(2)
with col1:
    st.header("Latest Weights")
    last_w = (
    weights.iloc[-1]                      # Series: index = tickers
           .rename("Weight")              # give the column a name
           .to_frame()                    # → DataFrame (ticker × Weight)
    )
    st.bar_chart(last_w)

with col2:
    st.header("Performance Stats")
    st.write(performance_stats(perf))

st.header("Equity Curve")
cum = (perf + 1).cumprod()
st.line_chart(cum)

# ---- DETAILS ----
st.header("Individual Stock Explorer")
sel = st.selectbox("Choose ticker", tickers)
st.line_chart(prices[sel])
st.write("Alpha scores for", sel)
st.line_chart(alpha.xs(sel, level="Ticker"))
