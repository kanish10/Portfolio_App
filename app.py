###############################################################################
#  Portfolio Coach  –  Dark-Mint Dashboard (Streamlit)
###############################################################################
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import altair as alt
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta
from streamlit_tags import st_tags

# Internal project helpers
from src.data_layer import fetch_prices
from src.features   import make_feature_panel
from src.signals    import composite_alpha
from src.portfolio  import weight_long_only, backtest
from src.utils      import performance_stats
###############################################################################
# 0. Page config
###############################################################################
st.set_page_config(
    page_title="🚀 Portfolio Coach",
    layout="wide",
    initial_sidebar_state="expanded",
)
###############################################################################
# 1. GLOBAL CSS (dark-mint) – fixes white header & font contrast
###############################################################################
st.markdown(
    """
    <style>
    /* paint the entire app */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], [data-testid="stToolbar"],
    [data-testid="stHeader"] { background:#0E1117 !important; }

    /* main block padding */
    .block-container{padding:1rem 2rem 2rem 2rem;background:#0E1117}

    /* left icon rail */
    .rail{position:fixed;top:0;bottom:0;left:0;width:54px;background:#1A1D27;z-index:999}
    .rail ul{list-style:none;margin:80px 0 0;padding:0}
    .rail li{width:54px;height:54px;display:flex;align-items:center;justify-content:center;
             transition:background .25s;cursor:pointer}
    .rail li:hover{background:#00CC96}
    .rail svg{stroke:#E0E0E0;width:22px;height:22px}
    .rail li:hover svg{stroke:#0E1117}

    /* cards */
    .card{background:#1F2330;border-radius:8px;padding:1.1rem 1.4rem;margin-bottom:1.6rem;
          box-shadow:0 1.5px 4px rgba(0,0,0,.45)}
    .card-header{font-size:1.2rem;font-weight:600;color:#E0E0E0;margin-bottom:.7rem}

    /* sidebar ticker box */
    .sidebar-card{background:#1F2330;border-radius:8px;padding:1rem;margin-bottom:1.3rem}
    .sidebar-card h4{margin:0;font-size:1.05rem;color:#E0E0E0;font-weight:500}

    /* widgets & sliders */
    textarea,select,input{background:#262730!important;color:#E0E0E0!important;
        border:1px solid #444F5A!important;border-radius:6px!important}
    .stSlider > div > div > div > div[role="slider"]{background:#00CC96!important}

    /* tag pills */
    .stTags input{background:#262730!important;color:#E0E0E0!important}
    .stTags .tagItem{
    background:#00CC96 !important;     /* mint by default            */
    border:1px solid #00CC96 !important;
    color:#0E1117  !important;
    border-radius:4px;
    padding:.15rem .45rem;
    margin:.1rem;
    }
    .stTags .tagItem:hover{
    background:#FFD700 !important;     /* optional: gold on hover    */
    border-color:#FFD700 !important;
    color:#0E1117 !important;
    }
    .stTags .removeTag{color:#FF4B4B!important;font-weight:bold;margin-left:.3rem}

    /* metrics */
    .stMetric>div[role="presentation"]{background:#262730!important;border:1px solid #444F5A!important;
        border-radius:8px;padding:.55rem .85rem}
    .stMetric p{color:#E0E0E0!important}
    .stMetric div>div>p{font-weight:600;font-size:1.25rem;color:#FFF!important}

    /* transparent charts */
    .stLineChart,.stBarChart,.stAltairChart{background:transparent!important}

    #MainMenu, footer{visibility:hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

# left icon rail
st.markdown(
    """
    <div class="rail">
      <ul>
        <li title="Signals"><svg viewBox="0 0 24 24"><path d="M4 19h16M4 12h10M4 5h6"/></svg></li>
        <li title="Stats"><svg viewBox="0 0 24 24"><path d="M4 19v-3m5 3V5m5 14v-7m5 7V9"/></svg></li>
        <li title="Charts"><svg viewBox="0 0 24 24"><path d="M3 3v18h18M7 16l3-4 4 5 5-7"/></svg></li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Altair dark-mint palette
def _theme():
    return {
        "config":{
            "background":"#0E1117",
            "axis":{"gridColor":"#222631","domainColor":"#444F5A",
                    "labelColor":"#888","titleColor":"#AAA"},
            "legend":{"labelColor":"#E0E0E0","titleColor":"#E0E0E0"},
            "title":{"color":"#E0E0E0"},
            "line":{"color":"#00CC96"},
            "range":{"category":["#00CC96","#FFD700","#FF7F50","#4AA8D8","#C997FF"]}
        }
    }
alt.themes.register("dark_mint", _theme); alt.themes.enable("dark_mint")
def _compound(s):
    """compound daily series into a 1-month return"""
    return (1.0 + s).prod() - 1.0
###############################################################################
# 2. helper: FF table (monthly)
###############################################################################
@st.cache_data(ttl=43200, show_spinner=False)
def load_or_build_merged(tickers):
    f = Path("data/merged_ff_data.parquet")
    if f.exists(): return pd.read_parquet(f)

    prices = fetch_prices(" ".join(tickers), start="2000-01-01")
    ret    = prices.pct_change().dropna()
    ff_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "F-F_Research_Data_Factors_daily_CSV.zip")
    raw = pd.read_csv(ff_url, skiprows=3, index_col=0, compression="zip")
    raw.index = pd.to_datetime(raw.index.astype(str), format="%Y%m%d", errors="coerce")
    ff = (raw[["Mkt-RF","SMB","HML","RF"]]
          .apply(pd.to_numeric, errors="coerce")
          .dropna().div(100))
    ff.columns = ["MktMinusRF","SMB","HML","RF"]

    comb = ret.join(ff, how="inner")
    ex   = comb[tickers].sub(comb["RF"], axis=0)
    long = (ex.stack()
              .reset_index()
              .rename(columns={"level_0":"Date","level_1":"Ticker",0:"ExcessReturn"}))
    factors = comb[["MktMinusRF","SMB","HML","RF"]].reset_index()
    df = long.merge(factors, on="Date")
    df["MonthEnd"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    df["OnePlus"]  = df["ExcessReturn"] + 1
    mret = (df.groupby(["MonthEnd","Ticker"])["OnePlus"]
              .prod().sub(1).reset_index(name="MonthlyRet"))
    
    
    mfct = (
    df.groupby("MonthEnd")[["MktMinusRF", "SMB", "HML", "RF"]]
      .agg(_compound)
      .reset_index()
        )
    fin  = mret.merge(mfct, on="MonthEnd")
    fin["ExcessReturn"] = fin["MonthlyRet"] - fin["RF"]
    fin = fin.rename(columns={"MonthEnd":"Date"})
    f.parent.mkdir(exist_ok=True); fin.to_parquet(f, index=False)
    return fin
###############################################################################
# 3.  SIDEBAR
###############################################################################
st.sidebar.markdown('<div class="sidebar-card"><h4>📋 Tickers</h4></div>', unsafe_allow_html=True)
_initial = st.session_state.get("user_tickers",
                               ["NVDA","AAPL","MSFT","C","GOOGL"])
tickers = st_tags(
    label       ="Add symbol ↵",
    text        ="e.g. TSLA ↵",
    value       =_initial,
    key         ="tick",
    maxtags     =30,
)
st.session_state["user_tickers"] = tickers

start  = st.sidebar.date_input("History from", date(2015,1,1),
                               min_value=date(2000,1,1))
q_long = st.sidebar.slider("Long-only quantile", .05,.5,.2,.05)
cutoff = st.sidebar.date_input("Train until", date.today()-timedelta(days=30))

if not tickers:
    st.stop()
###############################################################################
# 4.  Compute signals
###############################################################################
prices = fetch_prices(" ".join(tickers), start=start.isoformat())
if cutoff < date.today():
    prices = prices.loc[:pd.to_datetime(cutoff)]
panel   = make_feature_panel(prices)
alpha   = composite_alpha(panel)
weights = weight_long_only(alpha, q_long)
perf    = backtest(weights, prices).dropna()
###############################################################################
# 5.  Header
###############################################################################
st.markdown("## 🚀 Portfolio Coach")
st.markdown("<span style='color:#CCC'>Type tickers on the left, tune the "
            "slider, and explore the analytics below.</span>",
            unsafe_allow_html=True)
###############################################################################
# 6.  Cards
###############################################################################
def card(title, icon=""):
    st.markdown(f'<div class="card"><div class="card-header">{icon}&nbsp;{title}</div>',
                unsafe_allow_html=True)

# ── Signals ────────────────────────────────────────────────────────────────
card("Model Signals","⚡")
last = weights.index[-1]
longs  = weights.loc[last][weights.loc[last]>0].index.tolist()
shorts = [t for t in tickers if t not in longs]
st.markdown(f"<span style='color:#888'>As&nbsp;of&nbsp;{last:%Y-%m-%d}</span>",
            unsafe_allow_html=True)
st.markdown(f"<b style='color:#00CC96'>Buy/Add&nbsp;▶</b> "
            f"<span style='color:#E0E0E0'>{longs}</span>",
            unsafe_allow_html=True)
st.markdown(f"<b style='color:#FF4B4B'>Sell/Reduce&nbsp;▼</b> "
            f"<span style='color:#E0E0E0'>{shorts}</span>",
            unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Weights & stats ───────────────────────────────────────────────────────
card("Weights & Performance","📊")
c1,c2 = st.columns(2)
with c1:
    st.subheader("Latest Weights")
    show = weights.loc[last].rename("Weight").to_frame().sort_values("Weight",
                                                                     ascending=False)
    st.bar_chart(show)
with c2:
    st.subheader("Performance")
    s = performance_stats(perf)
    st.metric("CAGR",f"{s.CAGR:.1%}")
    st.metric("Volatility",f"{s.Vol:.1%}")
    st.metric("Sharpe",f"{s.Sharpe:.2f}")
    st.metric("Max DD",f"{s.MaxDD:.1%}")
st.markdown("</div>", unsafe_allow_html=True)

# ── Equity curve ──────────────────────────────────────────────────────────
card("Equity Curve vs Buy & Hold","📈")
cum = (perf+1).cumprod()
ew  = backtest(pd.DataFrame(1/len(tickers), index=perf.index, columns=tickers),
               prices).dropna().cumprod()
st.line_chart(pd.DataFrame({"Model":cum,"EW Buy & Hold":ew.reindex(cum.index).ffill()}))
st.markdown("</div>", unsafe_allow_html=True)

# ── Individual explorer ───────────────────────────────────────────────────
card("Individual Stock Explorer","🔎")
sel = st.selectbox("Choose ticker", tickers, index=0)
cl1,cl2 = st.columns(2)
with cl1:
    st.subheader(f"{sel} Price")
    st.line_chart(prices[sel])
with cl2:
    st.subheader(f"{sel} Alpha")
    st.line_chart(alpha.xs(sel, level="Ticker"))

###############################################################################
# 7.  Fama–French regression
###############################################################################
merged = load_or_build_merged(tickers)
df_t   = merged[merged["Ticker"]==sel].set_index("Date").loc[:pd.to_datetime(cutoff)]

if len(df_t)>=12:
    card("Fama–French 3-Factor Regression","🏛️")
    y  = df_t["ExcessReturn"]
    X  = sm.add_constant(df_t[["MktMinusRF","SMB","HML"]])
    mdl= sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":3})
    a,bm,bs,bv = mdl.params["const"],mdl.params["MktMinusRF"],mdl.params["SMB"],mdl.params["HML"]

    m1,m2,m3,m4=st.columns(4)
    m1.metric("α̂",f"{a:.4f}");   m2.metric("βₘ",f"{bm:.2f}")
    m3.metric("βₛ",f"{bs:.2f}");  m4.metric("βᵥ",f"{bv:.2f}")

    lastF = df_t[["MktMinusRF","SMB","HML"]].iloc[-1]
    pred  = a + bm*lastF["MktMinusRF"] + bs*lastF["SMB"] + bv*lastF["HML"]
    st.markdown(f"<span style='color:#CCC'>Predicted next-month excess: "
                f"<b>{pred:.2%}</b></span>", unsafe_allow_html=True)

    pts = df_t[["MktMinusRF","ExcessReturn"]].dropna()
    fit = pd.DataFrame({"x":[pts.MktMinusRF.min(),pts.MktMinusRF.max()],
                        "y":[a+bm*pts.MktMinusRF.min(),a+bm*pts.MktMinusRF.max()]})
    st.altair_chart(
        alt.layer(
            alt.Chart(pts.reset_index()).mark_circle(size=35,opacity=.6)
                .encode(x="MktMinusRF:Q", y="ExcessReturn:Q"),
            alt.Chart(fit).mark_line(strokeWidth=2,color="#FFD700")
                .encode(x="x:Q", y="y:Q")
        ).properties(height=260,width="container"),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

###############################################################################
# 8. Technical indicators
###############################################################################
card("Technical Indicators","🔧")
close = prices[sel]
ma50,ma200 = close.rolling(50).mean(), close.rolling(200).mean()
std20 = close.rolling(20).std()
upper,lower = ma50+2*std20, ma50-2*std20
delta = close.diff()
gain  = delta.where(delta>0,0); loss = -delta.where(delta<0,0)
rsi14 = 100 - 100/(1+gain.rolling(14).mean()/loss.rolling(14).mean())

df = (pd.DataFrame({"Close":close,"MA50":ma50,"MA200":ma200,
                    "UpperBB":upper,"LowerBB":lower})
      .reset_index())
base= alt.Chart(df.melt("Date")).encode(
        x="Date:T", y="value:Q",
        color=alt.Color("variable:N",
            scale=alt.Scale(domain=["Close","MA50","MA200","UpperBB","LowerBB"],
                            range=["#00CC96","#FFD700","#FF7F50","#00CC96","#00CC96"])))
st.altair_chart(
    base.transform_filter(alt.datum.variable=="Close").mark_line(size=2) +
    base.transform_filter(alt.datum.variable=="MA50").mark_line(strokeDash=[5,5]) +
    base.transform_filter(alt.datum.variable=="MA200").mark_line(strokeDash=[3,3]) +
    alt.Chart(df).mark_area(opacity=.08,color="#00CC96")
        .encode(x="Date:T", y="UpperBB:Q", y2="LowerBB:Q"),
    use_container_width=True
)
st.markdown("<span style='color:#CCC'>14-day RSI (overbought >70, "
            "oversold <30)</span>", unsafe_allow_html=True)
st.line_chart(rsi14.fillna(method="ffill"))
st.markdown("</div>", unsafe_allow_html=True)

###############################################################################
# 9. Footer
###############################################################################
st.markdown("<hr style='border-color:#222631'><p style='font-size:.85rem;color:#666'>"
            "Built by Kanish Khanna · Streamlit & Python · Data: Yahoo Finance"
            "</p>", unsafe_allow_html=True)
