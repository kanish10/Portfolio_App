import streamlit as st
import pandas as pd
import statsmodels.api as sm
import altair as alt
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta
from streamlit_tags import st_tags

from src.data_layer import fetch_prices
from src.features   import make_feature_panel
from src.signals    import composite_alpha
from src.portfolio  import weight_long_only, backtest
from src.utils      import performance_stats

def fetch_ff_factors() -> pd.DataFrame:
    """
    Download Ken French's daily FF 3-factor data, coerce the index to datetime,
    drop any non-date rows, convert percentages to decimals, and return a DataFrame
    with columns [MktMinusRF, SMB, HML, RF] indexed by a DatetimeIndex.
    """
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    )
    raw = pd.read_csv(
        url,
        header=0,
        index_col=0,
        parse_dates=False,
        skiprows=3,
        compression="zip",
    )
    try:
        dates = pd.to_datetime(raw.index.astype(str), format="%Y%m%d", errors="coerce")
    except:
        dates = pd.to_datetime(raw.index.astype(str), errors="coerce")
    raw.index = dates
    ff = raw.loc[raw.index.notna()]
    ff = ff.apply(pd.to_numeric, errors="coerce").dropna(how="all").astype(float).div(100)
    if list(ff.columns[:4]) == ["Mkt-RF", "SMB", "HML", "RF"]:
        ff.columns = ["MktMinusRF", "SMB", "HML", "RF"]
    else:
        ff = ff.iloc[:, :4]
        ff.columns = ["MktMinusRF", "SMB", "HML", "RF"]
    ff = ff.loc[ff.index.year >= 1900]
    return ff

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_or_build_merged(tickers: list[str]) -> pd.DataFrame:
    """
    If data/merged_ff_data.parquet exists, load & return it.
    Otherwise, download FF factors + prices, compute monthly merged table,
    save to Parquet, and return the result.
    """
    filepath = Path("data/merged_ff_data.parquet")
    if filepath.exists():
        return pd.read_parquet(filepath)

    prices = fetch_prices(" ".join(tickers), start="2000-01-01")
    prices.index = pd.to_datetime(prices.index)
    daily_ret = prices.pct_change().dropna()
    ff = fetch_ff_factors()
    combined = daily_ret.join(ff, how="inner")
    rf_series = combined["RF"]
    tickers_excess = combined[tickers].subtract(rf_series, axis=0)

    df_excess = (
        tickers_excess
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Date", "level_1": "Ticker", 0: "ExcessReturn"})
    )

    factors = (
        combined[["MktMinusRF", "SMB", "HML", "RF"]]
        .reset_index()
        .rename(columns={"index": "Date"})
    )

    df_long = pd.merge(df_excess, factors, on="Date", how="left")

    df_long["MonthEnd"] = df_long["Date"].dt.to_period("M").dt.to_timestamp("M")
    df_long["OnePlusEx"] = df_long["ExcessReturn"] + 1

    monthly_ret = (
        df_long
        .groupby(["MonthEnd", "Ticker"])["OnePlusEx"]
        .prod()
        .subtract(1)
        .reset_index(name="MonthlyRet")
    )

    factor_monthly = (
        df_long
        .groupby("MonthEnd")[["RF", "MktMinusRF", "SMB", "HML"]]
        .mean()
        .reset_index()
    )

    df_monthly = pd.merge(monthly_ret, factor_monthly, on="MonthEnd", how="left")
    df_monthly["ExcessReturn"] = df_monthly["MonthlyRet"] - df_monthly["RF"]

    final = df_monthly[[
        "MonthEnd", "Ticker", "ExcessReturn", "MktMinusRF", "SMB", "HML", "RF"
    ]].rename(columns={"MonthEnd": "Date"})

    filepath.parent.mkdir(exist_ok=True)
    final.to_parquet(filepath, index=False)
    return final

# ────────────────────────────────────────────────────────────────────────────
# 1) PAGE CONFIG & POLISHED DARK THEME (via config.toml + inline CSS)
st.set_page_config(
    page_title="🚀 Portfolio Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* HIDE BUILT-IN STREAMLIT UI */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* MAIN & SIDEBAR BACKGROUNDS */
    .css-18e3th9 {
        background-color: #0E1117 !important;
    }
    .css-1lcbmhc {
        background-color: #0E1117 !important;
    }
    .reportview-container .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    /* HEADERS & TEXT STYLING */
    h1, h2, h3, h4, h5 {
        color: #E0E0E0 !important;
        font-weight: 600;
    }
    p, span {
        color: #AAAAAA !important;
    }

    /* SIDEBAR INPUTS & TAGS */
    .css-1d391kg { color: #E0E0E0 !important; }
    .css-1u0ump2 textarea,
    .css-1v3fvcr select,
    .css-1nam4kp .stNumberInput>div>input {
        background-color: #1F2330 !important;
        color: #E0E0E0 !important;
        border-radius: 8px !important;
        border: 1px solid #444F5A !important;
        padding: 0.4rem 0.6rem !important;
    }
    .stSlider > div > div > div > div[role="slider"] {
        background-color: #00CC96 !important;
    }

    /* Tags generated by streamlit_tags */
    .stTags input {
        background-color: #1F2330 !important;
        color: #E0E0E0 !important;
    }
    .stTags .tagItem {
        background-color: #262730 !important;
        border: 1px solid #444F5A !important;
        color: #E0E0E0 !important;
        border-radius: 4px !important;
        padding: 0.2rem 0.5rem !important;
        margin: 0.1rem !important;
    }
    .stTags .tagItem:hover {
        background-color: #00CC96 !important;
        color: #0E1117 !important;
    }
    .stTags .tagItem .removeTag {
        color: #FF4B4B !important;
        font-weight: bold;
        margin-left: 0.3rem;
        cursor: pointer;
    }

    /* CHART & METRIC CARDS */
    .stLineChart, .stBarChart, .stAltairChart, .stPydeckChart {
        background-color: #1F2330 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    .stMetric > div[role="presentation"] {
        background-color: #1F2330 !important;
        border: 1px solid #444F5A !important;
        border-radius: 8px !important;
        padding: 0.7rem 1rem !important;
    }
    .stMetric > div[role="presentation"] > div > p {
        color: #E0E0E0 !important;
    }
    .stMetric > div[role="presentation"] > div > div > p {
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        color: #FFFFFF !important;
    }

    /* DATAFRAME STYLING */
    .stDataFrame .css-1g1b950 { background-color: transparent !important; }
    .stDataFrame th, td    { color: #E0E0E0 !important; }
    .stDataFrame th {
        border-bottom: 1px solid #444F5A !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR: “Portfolio Coach Settings” with TAG-STYLE TICKER ENTRY
st.sidebar.title("⚙️ Portfolio Coach Settings")

current = st.session_state.get("user_tickers", ["NVDA", "AAPL", "MSFT", "C", "GOOGL"])
selected_tickers = st_tags(
    label='Enter tickers and press Enter ↵',
    text='Type ticker (e.g. TSLA) and ↵',
    value=current,
    key='1',
    suggestions=[],
    maxtags=30,
)
st.session_state["user_tickers"] = selected_tickers
tickers = st.session_state["user_tickers"]

start_dt = st.sidebar.date_input(
    "History from", 
    value=date(2015, 1, 1),
    min_value=date(2000, 1, 1),
    help="Earliest data pull date"
)
top_cut = st.sidebar.slider(
    "Long-only quantile", 0.05, 0.5, 0.2, 0.05,
    help="Fraction of stocks to go long each month (top quantile)"
)
cutoff = st.sidebar.date_input(
    "Train until", 
    value=date.today() - timedelta(days=30),
    help="Use data up to this date for signal computation"
)
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 3) FETCH & PROCESS DATA based on tickers
if len(tickers) == 0:
    st.warning("➤ Please add at least one ticker in the sidebar to run the model.")
    st.stop()

prices = fetch_prices(" ".join(tickers), start=start_dt.isoformat())
if cutoff < date.today():
    prices = prices.loc[:pd.to_datetime(cutoff)]

panel = make_feature_panel(prices)
alpha = composite_alpha(panel)
weights = weight_long_only(alpha, top_cut)
perf = backtest(weights, prices).dropna()
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 4) MAIN APP LAYOUT
st.markdown("<h1>🚀 Portfolio Coach</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:1.1rem; color:#E0E0E0;'>"
    "Type tickers in the sidebar above, press ↵ to add. The model ranks them by composite alpha "
    "and suggests which to buy (top quantile) and which to sell."
    "</p>",
    unsafe_allow_html=True
)

latest_date = weights.index[-1]
st.markdown(f"### Model Signals for **{latest_date.date()}**")
latest_w = weights.loc[latest_date]
buy_list  = latest_w[latest_w > 0].index.tolist()
sell_list = [t for t in tickers if t not in buy_list]
st.markdown(f"<b style='color:#7FFF00;'>Buy/Add:</b> {buy_list}", unsafe_allow_html=True)
st.markdown(f"<b style='color:#FF4500;'>Sell/Reduce:</b> {sell_list}", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Latest Weights")
    last_w_df = latest_w.rename("Weight").to_frame().sort_values("Weight", ascending=False)
    st.bar_chart(last_w_df)
with col2:
    st.subheader("Performance Stats")
    stats = performance_stats(perf)
    st.metric("CAGR", f"{stats.CAGR:.1%}", delta=None)
    st.metric("Vol", f"{stats.Vol:.1%}", delta=None)
    st.metric("Sharpe", f"{stats.Sharpe:.2f}", delta=None)
    st.metric("Max Drawdown", f"{stats.MaxDD:.1%}", delta=None)

st.subheader("📈 Equity Curve vs. Benchmark (Buy & Hold)")
cum_model = (perf + 1).cumprod()
ew_w = pd.DataFrame(1/len(tickers), index=perf.index, columns=tickers)
ew_perf = backtest(ew_w, prices).dropna().cumprod()
df_equity = pd.DataFrame({
    "Model Portfolio": cum_model,
    "EW Buy & Hold":  ew_perf.reindex(cum_model.index).ffill()
})
st.line_chart(df_equity, use_container_width=True)
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 5) Individual Stock Explorer (Deep Dive)
st.subheader("🔎 Individual Stock Explorer")
sel = st.selectbox(
    "Choose ticker to drill into:",
    options=tickers,
    index=0,
    help="Select any ticker above to see price, alpha, FF regression, BS demo, technicals"
)

st.markdown(f"### {sel} – Price & Alpha History")
colA, colB = st.columns(2)
with colA:
    st.line_chart(prices[sel].rename("Price"), use_container_width=True)
with colB:
    alpha_sel = alpha.xs(sel, level="Ticker").sort_index()
    st.line_chart(alpha_sel, use_container_width=True)

merged = load_or_build_merged(tickers)
df_t   = merged[merged["Ticker"] == sel].set_index("Date")
df_fit = df_t.loc[:pd.to_datetime(cutoff)]

if len(df_fit) < 12:
    st.warning("Not enough monthly data to run a Fama–French regression for this ticker.")
else:
    y = df_fit["ExcessReturn"]
    X = sm.add_constant(df_fit[["MktMinusRF", "SMB", "HML"]])
    model     = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    beta_m    = model.params["MktMinusRF"]
    beta_s    = model.params["SMB"]
    beta_v    = model.params["HML"]
    alpha_hat = model.params["const"]

    st.markdown("<h3>Fama–French 3-Factor Regression</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("α̂", f"{alpha_hat:.4f}")
    c2.metric("βₘ", f"{beta_m:.2f}")
    c3.metric("βₛ", f"{beta_s:.2f}")
    c4.metric("βᵥ", f"{beta_v:.2f}")

    last_factors = df_t[["MktMinusRF", "SMB", "HML"]].iloc[-1]
    pred_excess  = alpha_hat + (
        beta_m * last_factors["MktMinusRF"] +
        beta_s * last_factors["SMB"] +
        beta_v * last_factors["HML"]
    )
    st.markdown(f"**Predicted Next-Month Excess Return:** {pred_excess:.2%}")

    chart_df = df_t[["MktMinusRF", "ExcessReturn"]].dropna()
    m, b = beta_m, alpha_hat
    scatter = alt.Chart(chart_df.reset_index()).mark_circle(size=40, color="#00CC96", opacity=0.6).encode(
        x=alt.X("MktMinusRF:Q", title="Market Excess (Mkt–RF)", axis=alt.Axis(format=".1%")),
        y=alt.Y("ExcessReturn:Q", title=f"{sel} Excess Return", axis=alt.Axis(format=".1%"))
    )
    reg_line = alt.Chart(pd.DataFrame({
        "MktMinusRF": [chart_df["MktMinusRF"].min(), chart_df["MktMinusRF"].max()],
        "Fit":        [b + m * chart_df["MktMinusRF"].min(), b + m * chart_df["MktMinusRF"].max()]
    })).mark_line(color="#FFFFFF", strokeWidth=2).encode(
        x="MktMinusRF:Q",
        y="Fit:Q"
    )
    st.altair_chart((scatter + reg_line).properties(height=300), use_container_width=True)

st.markdown("<h3>Black–Scholes Option Demo</h3>", unsafe_allow_html=True)
obj = yf.Ticker(sel)
exp_dates = obj.options
if exp_dates:
    nearest = exp_dates[0]
    opt_chain = obj.option_chain(nearest).calls
    S0 = prices[sel].iloc[-1]
    atm_row = opt_chain.iloc[(opt_chain["strike"] - S0).abs().idxmin()]
    K       = atm_row["strike"]
    mid_iv  = atm_row["impliedVolatility"]
    exp_date = pd.to_datetime(nearest)
    T = max((exp_date - prices.index[-1]).days, 1) / 365
    r = 0.02

    import numpy as np
    from scipy.stats import norm

    def bs_call_price(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    bs_price_val = bs_call_price(S0, K, T, r, mid_iv)
    st.write(f"Nearest exp: **{nearest}**  |  ATM strike: **{K:.2f}**")
    st.write(f"Spot S₀ = **{S0:.2f}**, IV = **{mid_iv:.2%}**,  T = **{T:.3f} yrs**,  r = **{r:.2%}**")
    st.write(f"BS theoretical call price: **${bs_price_val:.2f}**")
    st.write(f"Market last price for that call: **${atm_row['lastPrice']:.2f}**")
else:
    st.markdown("<p style='color:#AAAAAA;'>No options chain available for this ticker.</p>", unsafe_allow_html=True)

st.markdown("<h3>Technical Indicators</h3>", unsafe_allow_html=True)
close = prices[sel]
ma50  = close.rolling(50).mean()
ma200 = close.rolling(200).mean()
std20 = close.rolling(20).std()
upper_bb = ma50 + 2 * std20
lower_bb = ma50 - 2 * std20

delta    = close.diff()
gain     = delta.where(delta > 0, 0.0)
loss     = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
rsi14    = 100 - (100 / (1 + rs))

df_tech = pd.DataFrame({
    "Close":  close,
    "MA50":   ma50,
    "MA200":  ma200,
    "UpperBB": upper_bb,
    "LowerBB": lower_bb
}).reset_index().melt(id_vars="Date", var_name="Series", value_name="Value")

base = alt.Chart(df_tech).encode(
    x="Date:T",
    y=alt.Y("Value:Q", axis=alt.Axis(format="$~s")),
    color=alt.Color("Series:N", scale=alt.Scale(domain=["Close","MA50","MA200","UpperBB","LowerBB"],
                                               range=["#00CC96","#FFD700","#FF7F50","#00CC96","#00CC96"]))
)
price_line = base.transform_filter(alt.datum.Series == "Close").mark_line(size=2)
ma50_line  = base.transform_filter(alt.datum.Series == "MA50").mark_line(strokeDash=[5,5], strokeWidth=1.5)
ma200_line = base.transform_filter(alt.datum.Series == "MA200").mark_line(strokeDash=[3,3], strokeWidth=1.5)
bb_area    = alt.Chart(df_tech[df_tech["Series"].isin(["UpperBB","LowerBB"])]).mark_area(color="#00CC96", opacity=0.08).encode(
    x="Date:T",
    y="Value:Q",
    order=alt.Order("Series:N", sort="descending")
)
st.altair_chart((bb_area + price_line + ma50_line + ma200_line).properties(height=300), use_container_width=True)

st.markdown("<p style='color:#AAAAAA;'>14-day RSI (overbought >70, oversold <30)</p>", unsafe_allow_html=True)
st.line_chart(rsi14.fillna(method="ffill"), use_container_width=True)

st.markdown(
    "<hr><p style='font-size:0.9rem; color:#999;'>"
    "Built by Kanish Khanna · Powered by Streamlit & Python · Data from Yahoo Finance"
    "</p>",
    unsafe_allow_html=True
)
