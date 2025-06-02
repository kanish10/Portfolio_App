###############################################################################
#  Portfolio Coach  – Dark‐Mint Dashboard (Streamlit)
#
#  • Uses native Streamlit “multiselect” for tickers (no more streamlit_tags)
#  • Shows selected tickers as mint‐green “pills” in the sidebar
#  • MONTHLY Fama–French fix → correct monthly betas
#  • All CSS overrides for consistent dark/mint appearance
###############################################################################

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import altair as alt
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta

# ─── Internal project helpers (unchanged) ─────────────────────────────────
from src.data_layer import fetch_prices
from src.features    import make_feature_panel
from src.signals     import composite_alpha
from src.portfolio   import weight_long_only, backtest
from src.utils       import performance_stats

# ─── 0. Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Portfolio Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* ───────────────────────────────────────────────────────────────────── */
    /*   1) ENTIRE APP BACKGROUND (including header, toolbar, sidebar)       */
    /* ───────────────────────────────────────────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    [data-testid="stHeader"] {
      background: #0E1117 !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   2) MAIN BLOCK CONTAINER (padding + dark background)                */
    /* ───────────────────────────────────────────────────────────────────── */
    .block-container {
      padding: 1rem 2rem 2rem 2rem !important;
      background: #0E1117 !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   3) LEFT “ICON RAIL” (pure CSS for decorative sidebar icons)        */
    /* ───────────────────────────────────────────────────────────────────── */
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
      transition: background 0.25s;
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

    /* ───────────────────────────────────────────────────────────────────── */
    /*   4) CARD STYLING: slightly lighter dark (#1F2330) + subtle shadow   */
    /* ───────────────────────────────────────────────────────────────────── */
    .card {
      background: #1F2330 !important;
      border-radius: 8px !important;
      padding: 1.1rem 1.4rem !important;
      margin-bottom: 1.6rem !important;
      box-shadow: 0 1.5px 4px rgba(0, 0, 0, 0.45) !important;
    }
    .card-header {
      font-size: 1.2rem !important;
      font-weight: 600 !important;
      color: #E0E0E0 !important;
      margin-bottom: 0.7rem !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   5) SIDEBAR TICKER BOX (dark card)                                    */
    /* ───────────────────────────────────────────────────────────────────── */
    .sidebar-card {
      background: #1F2330 !important;
      border-radius: 8px !important;
      padding: 1rem !important;
      margin-bottom: 1.3rem !important;
    }
    .sidebar-card h4 {
      margin: 0 !important;
      font-size: 1.05rem !important;
      color: #E0E0E0 !important;
      font-weight: 500 !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   6) WIDGETS & SLIDERS (inputs, dropdowns, date pickers, sliders)     */
    /* ───────────────────────────────────────────────────────────────────── */
    textarea, select, input {
      background: #262730 !important;
      color: #E0E0E0 !important;
      border: 1px solid #444F5A !important;
      border-radius: 6px !important;
    }
    .stSlider > div > div > div > div[role="slider"] {
      background: #00CC96 !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   7) METRIC CARDS (Streamlit metrics)                                  */
    /* ───────────────────────────────────────────────────────────────────── */
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

    /* ───────────────────────────────────────────────────────────────────── */
    /*   8) CHART BACKGROUNDS: make all Streamlit charts transparent         */
    /* ───────────────────────────────────────────────────────────────────── */
    .stLineChart, .stBarChart, .stAltairChart {
      background: transparent !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   9) HIDE STREAMLIT FOOTER & MENU (they may appear white by default)  */
    /* ───────────────────────────────────────────────────────────────────── */
    #MainMenu, footer {
      visibility: hidden !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /* 10) MULTISELECT BOX  (dark background instead of white)               */
    /* ───────────────────────────────────────────────────────────────────── */
    /* Darken the outer Multiselect input field: */
    div[data-testid="stMultiSelect"] > div[role="combobox"] {
      background-color: #262730 !important;
      color: #E0E0E0 !important;
      border: 1px solid #444F5A !important;
      border-radius: 6px !important;
      padding: 4px !important;
    }
    /* Darken the inner area where you click/type: */
    div[data-testid="stMultiSelect"] input {
      background-color: #262730 !important;
      color: #E0E0E0 !important;
      padding: 4px !important;
    }
    /* Darken the dropdown menu area itself: */
    div[data-testid="stMultiSelect"] .css-1btezie { 
      background: #1F2330 !important;
      color: #E0E0E0 !important;
    }
    /* Make sure placeholder text in multiselect stays light grey */
    div[data-testid="stMultiSelect"] .css-1bdjhyg { 
      color: #888 !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /* 11) FORCE MINT-GREEN “PILL” STYLING (override any inline red/blue)    */
    /* ───────────────────────────────────────────────────────────────────── */

    /* 11.1) All <span data-baseweb="tag"> used by Streamlit (multi/select)  */
    span[data-baseweb="tag"] {
      background:      #00CC96 !important;
      background-color:#00CC96 !important;
      border-color:    #00CC96 !important;
      color:           #0E1117  !important;
      border-radius:   16px  !important;
      padding:         4px 12px !important;
      font-size:       0.875rem !important;
      font-weight:     500    !important;
      margin:          2px   !important;
      display:         inline-flex !important;
      align-items:     center !important;
      transition:      all 0.2s ease !important;
    }

    /* 11.2) Some versions wrap pills in a “.tagItem” <div>—override that too */
    .stTags .tagItem,
    div[data-testid="stTags"] .tagItem,
    .streamlit-tags .tagItem {
      background: #00CC96 !important;
      border-color: #00CC96 !important;
      color: #0E1117 !important;
      border-radius: 16px !important;
      padding: 4px 12px !important;
      font-size: 0.875rem !important;
      font-weight: 500 !important;
      margin: 2px !important;
      display: inline-flex !important;
      align-items: center !important;
      transition: all 0.2s ease !important;
    }

    /* 11.3) Nukes ANY inline “style=” that tries to turn a pill red, blue, etc. */
    span[data-baseweb="tag"][style],
    .stTags .tagItem[style],
    div[data-testid="stTags"] .tagItem[style],
    .streamlit-tags .tagItem[style] {
      background:      #00CC96 !important;
      background-color:#00CC96 !important;
      border-color:    #00CC96 !important;
      color:           #0E1117  !important;
    }

    /* 11.4) The “×” close-button on each pill: dark text on mint */
    span[data-baseweb="tag"] button,
    .stTags .removeTag,
    div[data-testid="stTags"] .removeTag,
    .streamlit-tags .tagItem button {
      color:       #0E1117 !important;
      background:  transparent   !important;
      font-weight: bold         !important;
      font-size:   14px         !important;
      line-height: 1           !important;
      border:      none        !important;
      cursor:      pointer     !important;
      opacity:     0.85        !important;
      transition:  opacity 0.2s ease !important;
    }
    /* 11.5) “×” hover effect (slightly brighter) */
    span[data-baseweb="tag"] button:hover,
    .stTags .removeTag:hover,
    .streamlit-tags .tagItem button:hover {
      opacity: 1 !important;
    }

    /* 11.6) Ensure the input field for adding new tags (if using st_tags) is dark */
    .stTags input,
    div[data-testid="stTags"] input,
    .streamlit-tags input {
      background: #262730 !important;
      color: #E0E0E0 !important;
      border: 1px solid #444F5A !important;
      border-radius: 6px !important;
      padding: 8px 12px !important;
    }
    /* 11.7) Wrap the entire tag container in a dark box, if it exists */
    .stTags > div,
    div[data-testid="stTags"] > div,
    .streamlit-tags > div {
      background: #1F2330 !important;
      border-radius: 8px !important;
      padding: 8px !important;
      border: 1px solid #444F5A !important;
    }

    /* ───────────────────────────────────────────────────────────────────── */
    /*   End of CSS block                                                     */
    /* ───────────────────────────────────────────────────────────────────── */
    </style>
    """,
    unsafe_allow_html=True
)


# ─── 2. Insert Left "icon rail" HTML ─────────────────────────────────────────
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
    unsafe_allow_html=True
)

# ─── 3. Altair "dark_mint" theme ────────────────────────────────────────────
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
            "category": ["#00CC96", "#FFD700", "#FF7F50", "#4AA8D8", "#C997FF"]
        }
    })

###############################################################################
# 4. BUILD MONTHLY Fama–French TABLE (fixed)
###############################################################################
@st.cache_data(ttl=43200, show_spinner=False)
def load_or_build_merged(tickers: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with MONTHLY:
      Date | Ticker | ExcessReturn | MktMinusRF | SMB | HML | RF

    • Uses Ken‐French's monthly factors (already in %-per‐month).
    • Calculates true monthly stock returns via resample('M').last().
    • Aligns returns & factors on the same Date index.
    """
    fp = Path("data/merged_ff_data.parquet")
    if fp.exists():
        return pd.read_parquet(fp)

    # 1) Daily → resample to month‐end closing prices
    prices = fetch_prices(" ".join(tickers), start="2000-01-01")
    monthly_prices = prices.resample("M").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    # 2) Download Ken‐French MONTHLY factors (already %/month)
    monthly_url = (
      "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
      "F-F_Research_Data_Factors_CSV.zip"
    )
    try:
        raw = pd.read_csv(monthly_url, compression="zip", skiprows=3, index_col=0)
        # Keep only rows with proper "YYYYMM" index:
        raw = raw[raw.index.astype(str).str.match(r"^\d{6}$")]
        dates = pd.to_datetime(raw.index.astype(str) + "01", format="%Y%m%d") \
                   .to_period("M").to_timestamp("M")

        ff = (
            raw[["Mkt-RF", "SMB", "HML", "RF"]]
              .set_index(dates)
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
              .div(100)  # Convert percent to decimal
        )
        ff.columns = ["MktMinusRF", "SMB", "HML", "RF"]
    except Exception as e:
        st.error(f"Error downloading Fama–French data: {e}")
        return pd.DataFrame()

    # 3) Align monthly stock returns and factors (‘inner’ join on same month‐end)
    aligned = monthly_returns.join(ff, how="inner")

    # 4) Compute stock excess returns: R_stock_monthly – RF_monthly
    stock_excess = aligned[tickers].sub(aligned["RF"], axis=0)

    # 5) Unpivot into long form, one row per (Date, Ticker)
    rows = []
    for tkr in tickers:
        df_tkr = pd.DataFrame({
            "Date":         stock_excess.index,
            "Ticker":       tkr,
            "ExcessReturn": stock_excess[tkr],
            "MktMinusRF":   aligned["MktMinusRF"],
            "SMB":          aligned["SMB"],
            "HML":          aligned["HML"],
            "RF":           aligned["RF"],
        })
        rows.append(df_tkr)

    final = pd.concat(rows, ignore_index=True).dropna()

    # 6) Save to Parquet & return
    fp.parent.mkdir(exist_ok=True)
    final.to_parquet(fp, index=False)
    return final

###############################################################################
# 5. Sidebar: Ticker entry & controls (multiselect + custom input)
###############################################################################
st.sidebar.markdown(
    '<div class="sidebar-card"><h4>📋 Tickers</h4></div>',
    unsafe_allow_html=True
)

# A pre‐defined list of “common” tickers for the multiselect dropdown
ALL_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","BRK-B","UNH","JNJ",
    "V","PG","JPM","HD","MA","DIS","PYPL","ADBE","CRM","NFLX","CMCSA",
    "XOM","VZ","T","KO","PFE","ABT","WMT","BAC","CSCO","INTC","TMO",
    "CVX","LLY","ABBV","ACN","NKE","MRK","COST","DHR","TXN","NEE",
    "C","GS","MS","AMGN","IBM","AMD","QCOM","SBUX","INTU","CAT"
]

_initial = st.session_state.get("user_tickers", ["NVDA","AAPL","MSFT","C","GOOGL"])
# Native multiselect
tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=ALL_TICKERS,
    default=_initial,
    key="tick"
)

# Additional free‐text input for any custom ticker not in the dropdown
custom_ticker = st.sidebar.text_input(
    "Add custom ticker:",
    placeholder="e.g. TSLA",
    key="custom_tick"
)

# If user enters a new custom ticker (and it isn't already in our list), add it
if custom_ticker:
    normalized = custom_ticker.strip().upper()
    if normalized and normalized not in tickers:
        if st.sidebar.button("Add Ticker"):
            tickers.append(normalized)
            # Force a rerun so that multiselect and pills update immediately
            st.experimental_rerun()

st.session_state["user_tickers"] = tickers

# Display the selected tickers as mint‐green pills below the multiselect
if tickers:
    pills_html = ""
    for tkr in tickers:
        pills_html += f"""
        <span style="
          display: inline-block;
          background: #00CC96;
          color: #0E1117;
          padding: 4px 12px;
          margin: 2px;
          border-radius: 16px;
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
        " onclick="
          // Find the “× Remove {tkr}” button in the multiselect and click it
          document
            .querySelector('[data-testid="stSidebar"] button[title="Remove {tkr}"]')
            ?.click();
        ">
          {tkr} ×
        </span>
        """
    st.sidebar.markdown(f"<div>**Selected:** {pills_html}</div>", unsafe_allow_html=True)

# If no tickers are selected, stop execution
if not tickers:
    st.sidebar.warning("➤ Please add at least one ticker above.")
    st.stop()

# Sidebar controls for date range and quantile
start  = st.sidebar.date_input("History from", date(2015,1,1), min_value=date(2000,1,1))
q_long = st.sidebar.slider("Long‐only quantile", 0.05, 0.5, 0.2, 0.05)
cutoff = st.sidebar.date_input("Train until", date.today() - timedelta(days=30))

###############################################################################
# 6. Fetch & compute model signals
###############################################################################
prices = fetch_prices(" ".join(tickers), start=start.isoformat())
if cutoff < date.today():
    prices = prices.loc[:pd.to_datetime(cutoff)]

panel   = make_feature_panel(prices)
alpha   = composite_alpha(panel)
weights = weight_long_only(alpha, q_long)
perf    = backtest(weights, prices).dropna()

###############################################################################
# 7. Header & Intro
###############################################################################
st.markdown("## 🚀 Portfolio Coach")
st.markdown(
    "<span style='color:#CCC'>Type tickers on the left, tune the slider, and explore the analytics below.</span>",
    unsafe_allow_html=True
)

###############################################################################
# 8. Convenience function for opening a “card”
###############################################################################
def card(title: str, icon: str="") -> None:
    """Open a <div class='card'>… block with a header."""
    st.markdown(
        f'<div class="card"><div class="card-header">{icon}&nbsp;{title}</div>',
        unsafe_allow_html=True
    )

# ── Model Signals ───────────────────────────────────────────────────────────
card("Model Signals", "⚡")
last    = weights.index[-1]
longs   = weights.loc[last][weights.loc[last] > 0].index.tolist()
shorts  = [t for t in tickers if t not in longs]

st.markdown(f"<span style='color:#888'>As of {last:%Y-%m-%d}</span>", unsafe_allow_html=True)
st.markdown(
    f"<b style='color:#00CC96'>Buy/Add ▶</b> <span style='color:#E0E0E0'>{longs}</span>",
    unsafe_allow_html=True
)
st.markdown(
    f"<b style='color:#FF4B4B'>Sell/Reduce ▼</b> <span style='color:#E0E0E0'>{shorts}</span>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Weights & Performance Stats ─────────────────────────────────────────────
card("Weights & Performance", "📊")
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

# ── Equity Curve vs Buy & Hold ──────────────────────────────────────────────
card("Equity Curve vs Buy & Hold", "📈")
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
    "Model Portfolio": cum,
    "EW Buy & Hold":   ew.reindex(cum.index).ffill()
})
st.line_chart(df_equity, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Individual Stock Explorer ───────────────────────────────────────────────
card("Individual Stock Explorer", "🔎")
sel = st.selectbox("Choose ticker", tickers, index=0)

colA, colB = st.columns(2)
with colA:
    st.subheader(f"{sel} Price")
    st.line_chart(prices[sel])
with colB:
    st.subheader(f"{sel} Alpha")
    st.line_chart(alpha.xs(sel, level="Ticker"))
st.markdown("</div>", unsafe_allow_html=True)

# ── Fama–French 3-Factor Regression (MONTHLY) ────────────────────────────────
merged = load_or_build_merged(tickers)

if not merged.empty:
    df_t = (
        merged[merged["Ticker"] == sel]
        .set_index("Date")
        .loc[:pd.to_datetime(cutoff)]
    )

    if len(df_t) >= 12:
        card("Fama–French 3-Factor Regression", "🏛️")

        # Define y and X in monthly units:
        y_data = df_t["ExcessReturn"].dropna()
        X_data = df_t.loc[y_data.index, ["MktMinusRF", "SMB", "HML"]]
        X      = sm.add_constant(X_data)

        try:
            mdl = sm.OLS(y_data, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            a  = mdl.params["const"]
            bm = mdl.params["MktMinusRF"]
            bs = mdl.params["SMB"]
            bv = mdl.params["HML"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("α̂ (monthly)", f"{a:.4f}")
            c2.metric("βₘ (market)", f"{bm:.2f}")
            c3.metric("βₛ (size)",   f"{bs:.2f}")
            c4.metric("βᵥ (value)",  f"{bv:.2f}")

            # Predicted next‐month excess:
            if len(X_data) > 0:
                lastF = X_data.iloc[-1]
                pred  = a + bm * lastF["MktMinusRF"] + bs * lastF["SMB"] + bv * lastF["HML"]
                st.markdown(
                    f"<span style='color:#CCC'>"
                    f"Predicted next-month excess: <b>{pred:.2%}</b>"
                    f"</span>",
                    unsafe_allow_html=True
                )

            # Scatter: mkt_excess vs stock_excess, with fitted line
            pts = pd.DataFrame({
                "MktMinusRF":   X_data["MktMinusRF"],
                "ExcessReturn": y_data
            }).dropna()

            if len(pts) > 1:
                fit = pd.DataFrame({
                    "x": [pts.MktMinusRF.min(), pts.MktMinusRF.max()],
                    "y": [a + bm * pts.MktMinusRF.min(), a + bm * pts.MktMinusRF.max()]
                })

                chart = alt.layer(
                    alt.Chart(pts.reset_index()).mark_circle(size=35, opacity=0.6).encode(
                        x=alt.X("MktMinusRF:Q", title="Mkt – RF (monthly)"),
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
card("Technical Indicators", "🔧")
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
rs       = avg_gain.div(avg_loss).fillna(0)
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
        domain=["Close", "MA50", "MA200", "UpperBB", "LowerBB"],
        range=["#00CC96", "#FFD700", "#FF7F50", "#00CC96", "#00CC96"]
    ))
)

chart_tech = (
    base.transform_filter(alt.datum.variable == "Close").mark_line(size=2) +
    base.transform_filter(alt.datum.variable == "MA50").mark_line(strokeDash=[5, 5]) +
    base.transform_filter(alt.datum.variable == "MA200").mark_line(strokeDash=[3, 3]) +
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
    "<p style='font-size:.85rem; color:#666'>"
    "Built by Kanish Khanna · Streamlit & Python · Data: Yahoo Finance"
    "</p>",
    unsafe_allow_html=True
)
