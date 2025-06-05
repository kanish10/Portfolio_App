###############################################################################
#  Portfolio Coach  –  Dark-Mint Dashboard (Streamlit)
#
#  1) FIXED: Monthly Fama‐French file → sensible betas (was mixing daily/monthly)
#  2) FIXED: Force mint‐green tag pills (better CSS targeting)
#  3) Remove any leftover red backgrounds on tags
###############################################################################

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import altair as alt
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta
from streamlit_tags import st_tags

# ─── Internal project helpers (no changes) ─────────────────────────────────
from src.data_layer import fetch_prices
from src.features   import make_feature_panel
from src.signals    import composite_alpha
from src.portfolio  import weight_long_only, backtest
from src.utils      import performance_stats

# ─── 0. Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Portfolio Coach",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 1. GLOBAL CSS (dark-mint) – FIXED tag styling ─────────────────────────
st.markdown(
    """
    <style>
    /* paint the entire app background (including header/toolbar) */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], [data-testid="stToolbar"],
    [data-testid="stHeader"] {
        background: #0E1117 !important;
    }

    /* main block container padding + background */
    .block-container {
        padding: 1rem 2rem 2rem 2rem;
        background: #0E1117 !important;
    }

    /* ─── Left "icon rail" (pure CSS) ────────────────────────────────────── */
    .rail {
        position: fixed;
        top: 0;
        bottom: 0;
        left: 0;
        width: 54px;
        background: #1A1D27;
        z-index: 999;
    }
    .rail ul {
        list-style: none;
        margin: 80px 0 0 0;
        padding: 0;
    }
    .rail li {
        width: 54px;
        height: 54px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background .25s;
        cursor: pointer;
    }
    .rail li:hover {
        background: #00CC96;
    }
    .rail svg {
        stroke: #E0E0E0;
        width: 22px;
        height: 22px;
    }
    .rail li:hover svg {
        stroke: #0E1117;
    }

    /* ─── Card styling ───────────────────────────────────────────────────── */
    .card {
        background: #1F2330;
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 1.5px 4px rgba(0,0,0,0.45);
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #E0E0E0;
        margin-bottom: 0.7rem;
    }

    /* ─── Sidebar ticker box ────────────────────────────────────────────── */
    .sidebar-card {
        background: #1F2330;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.3rem;
    }
    .sidebar-card h4 {
        margin: 0;
        font-size: 1.05rem;
        color: #E0E0E0;
        font-weight: 500;
    }

    /* ─── Widgets & sliders ────────────────────────────────────────────── */
    textarea, select, input {
        background: #262730 !important;
        color: #E0E0E0 !important;
        border: 1px solid #444F5A !important;
        border-radius: 6px !important;
    }
    .stSlider > div > div > div > div[role="slider"] {
        background: #00CC96 !important;
    }

    /* ─── FIXED: Tag pills styling (more comprehensive targeting) ────────── */
    /* Input field for tags */
    .stTags input, 
    div[data-testid="stTags"] input {
        background: #262730 !important;
        color: #E0E0E0 !important;
        border: 1px solid #444F5A !important;
    }
    
    /* Tag items - target multiple possible selectors */
    .stTags .tagItem,
    div[data-testid="stTags"] .tagItem,
    .stTags span[data-baseweb="tag"],
    div[data-testid="stTags"] span[data-baseweb="tag"],
    .streamlit-tags .tagItem,
    span[data-baseweb="tag"] {
        background: #00CC96 !important;
        border: 1px solid #00CC96 !important;
        color: #0E1117 !important;
        border-radius: 4px !important;
        padding: 0.15rem 0.45rem !important;
        margin: 0.1rem !important;
        display: inline-flex !important;
        align-items: center !important;
        font-weight: 500 !important;
    }
    
    /* Force override any inline styles on tags */
    .stTags .tagItem[style],
    div[data-testid="stTags"] .tagItem[style],
    span[data-baseweb="tag"][style] {
        background: #00CC96 !important;
        border-color: #00CC96 !important;
        color: #0E1117 !important;
    }
    
    /* Remove/close button on tags */
    .stTags .removeTag,
    .stTags .tagItem button,
    div[data-testid="stTags"] .removeTag,
    div[data-testid="stTags"] .tagItem button,
    span[data-baseweb="tag"] button {
        background: transparent !important;
        color: #0E1117 !important;
        border: none !important;
        font-weight: bold !important;
        margin-left: 0.3rem !important;
        cursor: pointer !important;
    }

    /* ─── Metric cards ─────────────────────────────────────────────────── */
    .stMetric > div[role="presentation"] {
        background: #262730 !important;
        border: 1px solid #444F5A !important;
        border-radius: 8px !important;
        padding: 0.55rem 0.85rem !important;
    }
    .stMetric p {
        color: #E0E0E0 !important;
    }
    .stMetric div > div > p {
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        color: #FFFFFF !important;
    }

    /* ─── Make all charts transparent (card's bg shows through) ───────── */
    .stLineChart, .stBarChart, .stAltairChart {
        background: transparent !important;
    }

    /* hide Streamlit's default menu + footer (they sometimes appear white) */
    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Left "icon rail" insertion ─────────────────────────────────────────────
st.markdown(
    """
    <div class="rail">
      <ul>
        <li title="Signals">
          <svg viewBox="0 0 24 24"><path d="M4 19h16M4 12h10M4 5h6"/></svg>
        </li>
        <li title="Stats">
          <svg viewBox="0 0 24 24"><path d="M4 19v-3m5 3V5m5 14v-7m5 7V9"/></svg>
        </li>
        <li title="Charts">
          <svg viewBox="0 0 24 24"><path d="M3 3v18h18M7 16l3-4 4 5 5-7"/></svg>
        </li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Altair "dark_mint" theme (FIXED deprecation warning) ────────────────────
@alt.theme.register("dark_mint", enable=True)
def _theme():
    return alt.theme.ThemeConfig({
        "background": "#0E1117",
        "axis": {
            "gridColor": "#222631",
            "domainColor": "#444F5A",
            "labelColor": "#888",
            "titleColor": "#AAA"
        },
        "legend": {"labelColor": "#E0E0E0", "titleColor": "#E0E0E0"},
        "title": {"color": "#E0E0E0"},
        "line": {"color": "#00CC96"},
        "range": {
            "category": ["#00CC96","#FFD700","#FF7F50","#4AA8D8","#C997FF"]
        }
    })

# ─── 2. FIXED load_or_build_merged() (proper monthly data handling) ─────────
@st.cache_data(ttl=43200, show_spinner=False)
def load_or_build_merged(tickers: list[str]) -> pd.DataFrame:
    """
    FIXED VERSION: Produces a monthly table of:
      Date | Ticker | ExcessReturn | MktMinusRF | SMB | HML | RF

    • Uses Ken French's monthly F-F factors (already %-per-month)
    • Converts stock prices to monthly returns properly
    • Both stock returns and factors are now in same units (monthly %)
    """
    fp = Path("data/merged_ff_data.parquet")
    if fp.exists():
        return pd.read_parquet(fp)

    # 1) Fetch daily prices, convert to monthly prices (month-end)
    prices = fetch_prices(" ".join(tickers), start="2000-01-01")
    # Resample to month-end prices
    monthly_prices = prices.resample('M').last()
    # Calculate monthly returns
    monthly_returns = monthly_prices.pct_change().dropna()

    # 2) Download Ken French MONTHLY factors (already in % per month)
    monthly_url = (
      "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
      "F-F_Research_Data_Factors.CSV.zip"
    )
    try:
        raw = pd.read_csv(monthly_url, compression="zip", skiprows=3, index_col=0)
        # Handle the date conversion more robustly
        raw = raw[raw.index.astype(str).str.len() == 6]  # Filter valid YYYYMM format
        dates = pd.to_datetime(raw.index.astype(str) + "01", format="%Y%m%d").to_period("M").to_timestamp("M")
        
        ff = (raw[["Mkt-RF","SMB","HML","RF"]]
                .set_index(dates)
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .div(100))  # Convert from % to decimal
        ff.columns = ["MktMinusRF","SMB","HML","RF"]
    except Exception as e:
        st.error(f"Error downloading Fama-French data: {e}")
        # Return empty DataFrame or use dummy data
        return pd.DataFrame()

    # 3) Align monthly stock returns with monthly factors
    # Both are now monthly frequency
    aligned_data = monthly_returns.join(ff, how='inner')
    
    # 4) Calculate monthly excess returns for stocks
    stock_excess = aligned_data[tickers].sub(aligned_data["RF"], axis=0)

    # 5) Reshape to long format
    results = []
    for ticker in tickers:
        ticker_data = pd.DataFrame({
            'Date': stock_excess.index,
            'Ticker': ticker,
            'ExcessReturn': stock_excess[ticker],
            'MktMinusRF': aligned_data['MktMinusRF'],
            'SMB': aligned_data['SMB'],
            'HML': aligned_data['HML'],
            'RF': aligned_data['RF']
        })
        results.append(ticker_data)
    
    final = pd.concat(results, ignore_index=True).dropna()

    # 6) Save to Parquet & return
    fp.parent.mkdir(exist_ok=True)
    final.to_parquet(fp, index=False)
    return final

# ─── 3. Sidebar: Ticker entry & controls ─────────────────────────────────────
st.sidebar.markdown(
    '<div class="sidebar-card"><h4>📋 Tickers</h4></div>',
    unsafe_allow_html=True
)

_initial = st.session_state.get("user_tickers", ["NVDA","AAPL","MSFT","C","GOOGL"])
tickers = st_tags(
    label       ="Add symbol ↵",
    text        ="e.g. TSLA ↵",
    value       =_initial,
    key         ="tick",
    maxtags     =30
)
st.session_state["user_tickers"] = tickers

start  = st.sidebar.date_input("History from", date(2015,1,1), min_value=date(2000,1,1))
q_long = st.sidebar.slider("Long-only quantile", 0.05, 0.5, 0.2, 0.05)
cutoff = st.sidebar.date_input("Train until", date.today() - timedelta(days=30))

if not tickers:
    st.sidebar.warning("➤ Please add at least one ticker above.")
    st.stop()

# ─── 4. Fetch & build signals ─────────────────────────────────────────────────
prices = fetch_prices(" ".join(tickers), start=start.isoformat())
if cutoff < date.today():
    prices = prices.loc[:pd.to_datetime(cutoff)]

panel   = make_feature_panel(prices)
alpha   = composite_alpha(panel)
weights = weight_long_only(alpha, q_long)
perf    = backtest(weights, prices).dropna()

# ─── 5. Header & Intro ───────────────────────────────────────────────────────
st.markdown("## 🚀 Portfolio Coach")
st.markdown(
    "<span style='color:#CCC'>Type tickers on the left, tune the slider, "
    "and explore the analytics below.</span>",
    unsafe_allow_html=True
)

# │───────────────────────────────────────────────────────────────────────────│
# │ 6. Cards (Signals, Weights, Equity, Stock Explorer, Fama–French, Technical) │
# │───────────────────────────────────────────────────────────────────────────│

def card(title: str, icon: str="") -> None:
    """Convenience wrapper: open a <div class='card'>… block."""
    st.markdown(
        f'<div class="card">'
        f'  <div class="card-header">{icon}&nbsp;{title}</div>',
        unsafe_allow_html=True
    )

# ── Model Signals ────────────────────────────────────────────────────────────
card("Model Signals","⚡")
last = weights.index[-1]
longs  = weights.loc[last][weights.loc[last] > 0].index.tolist()
shorts = [t for t in tickers if t not in longs]

st.markdown(
    f"<span style='color:#888'>As of {last:%Y-%m-%d}</span>",
    unsafe_allow_html=True
)
st.markdown(
    f"<b style='color:#00CC96'>Buy/Add ▶</b> "
    f"<span style='color:#E0E0E0'>{longs}</span>",
    unsafe_allow_html=True
)
st.markdown(
    f"<b style='color:#FF4B4B'>Sell/Reduce ▼</b> "
    f"<span style='color:#E0E0E0'>{shorts}</span>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Weights & Performance Stats ─────────────────────────────────────────────
card("Weights & Performance","📊")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Latest Weights")
    show = (
        weights.loc[last]
               .rename("Weight")
               .to_frame()
               .sort_values("Weight", ascending=False)
    )
    st.bar_chart(show)

with col2:
    st.subheader("Performance")
    s = performance_stats(perf)
    st.metric("CAGR", f"{s.CAGR:.1%}")
    st.metric("Volatility", f"{s.Vol:.1%}")
    st.metric("Sharpe", f"{s.Sharpe:.2f}")
    st.metric("Max DD", f"{s.MaxDD:.1%}")
st.markdown("</div>", unsafe_allow_html=True)

# ── Equity Curve vs Buy & Hold ───────────────────────────────────────────────
card("Equity Curve vs Buy & Hold","📈")
cum = (perf + 1).cumprod()
ew  = (
    backtest(
        pd.DataFrame(1/len(tickers), index=perf.index, columns=tickers),
        prices
    )
    .dropna()
    .cumprod()
)
df_equity = pd.DataFrame({
    "Model": cum,
    "EW Buy & Hold": ew.reindex(cum.index).ffill()
})
st.line_chart(df_equity, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Individual Stock Explorer ────────────────────────────────────────────────
card("Individual Stock Explorer","🔎")
sel = st.selectbox("Choose ticker", tickers, index=0)

colA, colB = st.columns(2)
with colA:
    st.subheader(f"{sel} Price")
    st.line_chart(prices[sel])
with colB:
    st.subheader(f"{sel} Alpha")
    st.line_chart(alpha.xs(sel, level="Ticker"))
st.markdown("</div>", unsafe_allow_html=True)

# ── FIXED Fama–French 3-Factor Regression ───────────────────────────────────
merged = load_or_build_merged(tickers)

if not merged.empty:
    df_t = (
        merged[merged["Ticker"] == sel]
        .set_index("Date")
        .loc[:pd.to_datetime(cutoff)]
    )

    if len(df_t) >= 12:
        card("Fama–French 3-Factor Regression","🏛️")
        
        # Now both y and X are in proper monthly units
        y = df_t["ExcessReturn"].dropna()
        X_data = df_t[["MktMinusRF","SMB","HML"]].loc[y.index]
        X = sm.add_constant(X_data)
        
        try:
            mdl = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

            a  = mdl.params["const"]
            bm = mdl.params["MktMinusRF"]
            bs = mdl.params["SMB"]
            bv = mdl.params["HML"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("α̂ (monthly)", f"{a:.4f}")
            c2.metric("βₘ (market)", f"{bm:.2f}")
            c3.metric("βₛ (size)", f"{bs:.2f}")
            c4.metric("βᵥ (value)", f"{bv:.2f}")

            # Predicted next-month excess using latest factor values
            if len(X_data) > 0:
                lastF = X_data.iloc[-1]
                pred = a + bm * lastF["MktMinusRF"] + bs * lastF["SMB"] + bv * lastF["HML"]
                st.markdown(
                    f"<span style='color:#CCC'>"
                    f"Predicted next-month excess: <b>{pred:.2%}</b>"
                    f"</span>",
                    unsafe_allow_html=True
                )

            # Scatter plot: Market excess return vs Stock excess return
            pts = pd.DataFrame({
                'MktMinusRF': X_data["MktMinusRF"],
                'ExcessReturn': y
            }).dropna()
            
            if len(pts) > 1:
                fit = pd.DataFrame({
                    "x": [pts.MktMinusRF.min(), pts.MktMinusRF.max()],
                    "y": [a + bm * pts.MktMinusRF.min(), a + bm * pts.MktMinusRF.max()]
                })

                chart = alt.layer(
                    alt.Chart(pts.reset_index()).mark_circle(size=35, opacity=0.6).encode(
                        x=alt.X("MktMinusRF:Q", title="Market Excess Return"),
                        y=alt.Y("ExcessReturn:Q", title=f"{sel} Excess Return")
                    ),
                    alt.Chart(fit).mark_line(strokeWidth=2, color="#FFD700").encode(
                        x="x:Q",
                        y="y:Q"
                    )
                ).properties(height=260, width="container")

                st.altair_chart(chart, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in regression: {e}")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ── Technical Indicators ────────────────────────────────────────────────────
card("Technical Indicators","🔧")
close   = prices[sel]
ma50    = close.rolling(50).mean()
ma200   = close.rolling(200).mean()
std20   = close.rolling(20).std()
upperBB = ma50 + 2 * std20
lowerBB = ma50 - 2 * std20

delta    = close.diff()
gain     = delta.where(delta > 0, 0.0)
loss     = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
rsi14    = 100 - (100 / (1 + rs))

df_tech = pd.DataFrame({
    "Close":   close,
    "MA50":    ma50,
    "MA200":   ma200,
    "UpperBB": upperBB,
    "LowerBB": lowerBB
}).reset_index()

base = alt.Chart(df_tech.melt("Date")).encode(
    x="Date:T",
    y="value:Q",
    color=alt.Color("variable:N", scale=alt.Scale(
        domain=["Close","MA50","MA200","UpperBB","LowerBB"],
        range=["#00CC96","#FFD700","#FF7F50","#00CC96","#00CC96"]
    ))
)

chart_tech = (
    base.transform_filter(alt.datum.variable == "Close").mark_line(size=2) +
    base.transform_filter(alt.datum.variable == "MA50").mark_line(strokeDash=[5,5]) +
    base.transform_filter(alt.datum.variable == "MA200").mark_line(strokeDash=[3,3]) +
    alt.Chart(df_tech).mark_area(opacity=0.08, color="#00CC96").encode(
        x="Date:T",
        y="UpperBB:Q",
        y2="LowerBB:Q"
    )
).properties(height=260, width="container")

st.altair_chart(chart_tech, use_container_width=True)
st.markdown(
    "<span style='color:#CCC'>14-day RSI (Overbought >70, Oversold <30)</span>",
    unsafe_allow_html=True
)
st.line_chart(rsi14.ffill())

st.markdown("</div>", unsafe_allow_html=True)

# ─── 9. Footer ─────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:#222631'>"
    "<p style='font-size:.85rem;color:#666'>"
    "Built by Kanish Khanna · Streamlit & Python · Data: Yahoo Finance"
    "</p>",
    unsafe_allow_html=True
)