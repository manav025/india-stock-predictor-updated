import streamlit as st
import torch
import torch.nn as nn
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from newsapi import NewsApiClient

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. PAGE CONFIG — must be the absolute first Streamlit call  ║
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(page_title="AI Quant Terminal", page_icon="📈", layout="wide")


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. NEURAL NETWORK ARCHITECTURE                              ║
# ╚══════════════════════════════════════════════════════════════╝
class MasterQuantLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. LOAD MODEL & SCALER (cached — runs only once)            ║
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_resource
def load_ai_engine():
    device = torch.device("cpu")
    model  = MasterQuantLSTM().to(device)
    model.load_state_dict(torch.load("macro_lstm_master.pth", map_location=device))
    model.eval()
    scaler = joblib.load("master_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_ai_engine()
except FileNotFoundError:
    st.error("⚠️ Model files missing! Upload 'macro_lstm_master.pth' and 'master_scaler.pkl'.")
    st.stop()


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. CONSTANTS                                                ║
# ╚══════════════════════════════════════════════════════════════╝
SEQ_LEN = 5   # Must match training sequence length

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'sentiment',
    'sma_5', 'sma_20', 'rsi_14', 'std_20',
    'bollinger_high', 'bollinger_low',
    'vix', 'oil', 'usd_inr'
]

AVAILABLE_STOCKS = sorted([
    "ADANIENT.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "CANBK.NS",
    "COALINDIA.NS", "DRREDDY.NS", "GRASIM.NS", "HAL.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "MM.NS", "MARUTI.NS", "NTPC.NS", "ONGC.NS",
    "PNB.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "VEDL.NS",
    "WIPRO.NS", "ZOMATO.NS",
])


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. FEATURE ENGINEERING                                      ║
# ╚══════════════════════════════════════════════════════════════╝
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma_5']  = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()

    delta = df['close'].diff()
    gain  = delta.where(delta > 0,  0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df['std_20']         = df['close'].rolling(20).std()
    df['bollinger_high'] = df['sma_20'] + df['std_20'] * 2
    df['bollinger_low']  = df['sma_20'] - df['std_20'] * 2

    return df.dropna()


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. CACHED DATA FETCHERS  (TTL = 1 hour)                     ║
# ╚══════════════════════════════════════════════════════════════╝
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make the DatetimeIndex timezone-naive and date-only."""
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)   # safe even if already naive
    df.index = df.index.normalize()
    return df

def _yf_download_with_retry(tickers: list, period: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Single batched yf.download() call with exponential backoff.
    One HTTP request for N tickers beats N separate requests every time —
    far less likely to trigger Yahoo Finance rate limits.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,      # sequential inside yfinance — safer on shared IPs
            )
            if not raw.empty:
                return raw
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)   # 1 s → 2 s → 4 s
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str):
    raw = _yf_download_with_retry([ticker], period)
    if raw.empty:
        return None
    # Single-ticker download: columns are flat (Open, High, …)
    # Multi-ticker download: columns are MultiIndex — unwrap if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(ticker, axis=1, level=1)
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return _normalize_index(df)


@st.cache_data(ttl=3600)
def fetch_macro_data(period: str):
    """
    Downloads VIX, Crude Oil, and USD/INR in ONE batched request.
    Falls back gracefully if any single series is missing.
    """
    MACRO_MAP = {
        "^INDIAVIX": "vix",
        "CL=F":      "oil",
        "INR=X":     "usd_inr",
    }
    raw = _yf_download_with_retry(list(MACRO_MAP.keys()), period)
    if raw.empty:
        st.error("❌ Macro data unavailable after retries. Check your connection.")
        return None

    # yf.download with multiple tickers returns MultiIndex columns: (field, ticker)
    close = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]

    series = {}
    for yf_ticker, col_name in MACRO_MAP.items():
        if yf_ticker in close.columns:
            series[col_name] = close[yf_ticker]
        else:
            st.warning(f"⚠️ {col_name.upper()} ({yf_ticker}) missing from batch — skipping.")

    if len(series) < len(MACRO_MAP):
        st.error("❌ One or more macro series unavailable. Cannot build full feature set.")
        return None

    macro = pd.DataFrame(series).ffill().dropna()
    return _normalize_index(macro)

@st.cache_data(ttl=3600)
def fetch_analyst_target(symbol: str, api_key: str) -> str:
    try:
        url = f"https://stock.indianapi.in/stock_target_price?stock_name={symbol}"
        res = requests.get(url, headers={"x-api-key": api_key}, timeout=10).json()
        if isinstance(res, list) and res:
            return res[0].get("targetPrice", "N/A")
    except Exception:
        pass
    return "N/A"


@st.cache_data(ttl=1800)   # 30-min cache — news changes faster than prices
def get_sentiment_score(symbol: str, news_api_key: str) -> tuple[float, str]:
    """
    Fetches the latest 15 headlines for a stock via NewsAPI and returns:
      - polarity score  : float in [-1.0, +1.0]
      - label           : 'Bullish' | 'Bearish' | 'Neutral'

    TextBlob's PatternAnalyzer is used (no NLTK downloads needed).
    Only applied to the live prediction window; historical rows keep 0.0
    because free-tier NewsAPI doesn't provide old articles.
    """
    try:
        client   = NewsApiClient(api_key=news_api_key)
        response = client.get_everything(
            q=symbol,
            language='en',
            sort_by='publishedAt',
            page_size=15
        )
        articles = response.get('articles', [])
        if not articles:
            return 0.0, "Neutral"

        scores = [
            TextBlob(a['title']).sentiment.polarity
            for a in articles
            if a.get('title')
        ]
        if not scores:
            return 0.0, "Neutral"

        avg = float(np.mean(scores))
        if   avg >  0.05: label = "🟢 Bullish"
        elif avg < -0.05: label = "🔴 Bearish"
        else:             label = "⚪ Neutral"
        return avg, label

    except Exception:
        return 0.0, "Neutral"


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. AI PREDICTION                                            ║
# ╚══════════════════════════════════════════════════════════════╝
def run_prediction(df: pd.DataFrame) -> float | None:
    window = df.tail(SEQ_LEN)
    if len(window) < SEQ_LEN:
        return None
    scaled = scaler.transform(window[FEATURE_COLS].values)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(tensor)).item() * 100


# ╔══════════════════════════════════════════════════════════════╗
# ║  8. BACKTESTING ENGINE                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def run_backtest(df: pd.DataFrame, buy_thresh: float, sell_thresh: float):
    records = []
    for i in range(SEQ_LEN, len(df)):
        window = df.iloc[i - SEQ_LEN:i]
        try:
            scaled = scaler.transform(window[FEATURE_COLS].values)
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item() * 100

            actual_ret = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]

            if   prob > buy_thresh:  signal, strat_ret = "BUY",  actual_ret
            elif prob < sell_thresh: signal, strat_ret = "SELL", -actual_ret
            else:                    signal, strat_ret = "HOLD",  0.0

            records.append({
                'date': df.index[i], 'close': df['close'].iloc[i],
                'prob': prob, 'signal': signal,
                'actual_ret': actual_ret, 'strat_ret': strat_ret
            })
        except Exception:
            continue

    if not records:
        return None

    bt = pd.DataFrame(records).set_index('date')
    bt['cum_strategy'] = (1 + bt['strat_ret']).cumprod()
    bt['cum_buyhold']  = (1 + bt['actual_ret']).cumprod()
    return bt


# ╔══════════════════════════════════════════════════════════════╗
# ║  9. PLOTLY CHARTS                                            ║
# ╚══════════════════════════════════════════════════════════════╝
_DARK = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
             font=dict(color='#e0e0e0'))
_GRID = dict(gridcolor='rgba(255,255,255,0.08)')


def chart_technical(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=(f"{ticker} — Price & Bollinger Bands", "RSI (14)", "Volume"),
    )

    # — Candlestick —
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price",
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # — Bollinger Bands —
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bollinger_high'], name="BB High",
        line=dict(color='rgba(255,165,0,0.6)', dash='dash'), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bollinger_low'], name="BB Low",
        line=dict(color='rgba(255,165,0,0.6)', dash='dash'),
        fill='tonexty', fillcolor='rgba(255,165,0,0.04)', showlegend=False
    ), row=1, col=1)

    # — SMA 20 —
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma_20'], name="SMA 20",
        line=dict(color='rgba(100,149,237,0.8)', width=1.2), showlegend=False
    ), row=1, col=1)

    # — RSI —
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi_14'], name="RSI",
        line=dict(color='#ab47bc', width=1.5)
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.6)",  row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.6)", row=2, col=1)

    # — Volume bars —
    bar_colors = [
        '#26a69a' if c >= o else '#ef5350'
        for c, o in zip(df['close'], df['open'])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], name="Volume",
        marker_color=bar_colors, showlegend=False
    ), row=3, col=1)

    fig.update_layout(
        height=620, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        **_DARK
    )
    for axis in ['yaxis', 'yaxis2', 'yaxis3']:
        fig.update_layout(**{axis: _GRID})
    return fig


def chart_backtest(bt: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt['cum_strategy'],
        name="AI Strategy", line=dict(color='#26a69a', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt['cum_buyhold'],
        name="Buy & Hold", line=dict(color='#ef5350', width=2, dash='dash')
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"{ticker} — AI Strategy vs Buy & Hold",
        yaxis_title="Portfolio Growth (1.0 = no change)",
        height=420, yaxis=_GRID, **_DARK
    )
    return fig


def chart_correlation(tickers: list, period: str):
    returns = {}
    for t in tickers:
        df = fetch_stock_data(t, period)
        if df is not None:
            returns[t.replace('.NS', '')] = df['close'].pct_change().dropna()

    if len(returns) < 2:
        return None

    corr = pd.DataFrame(returns).dropna().corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Pearson r")
    ))
    fig.update_layout(
        title="Return Correlation Matrix",
        height=max(420, len(returns) * 65),
        **_DARK
    )
    return fig


# ╔══════════════════════════════════════════════════════════════╗
# ║  10. UI — HEADER & SIDEBAR                                   ║
# ╚══════════════════════════════════════════════════════════════╝
st.title("📈 Institutional AI Trading Terminal")
st.markdown("Powered by Deep Learning & IndianAPI")

st.sidebar.header("⚙️ System Configuration")

try:
    user_api_key = st.secrets["INDIAN_API_KEY"]
    st.sidebar.success("✅ API Key: CONNECTED (Securely)")
except Exception:
    st.sidebar.error("🚨 API Key missing in Streamlit Secrets!")
    user_api_key = None

try:
    news_api_key = st.secrets["NEWS_API_KEY"]
    st.sidebar.success("✅ Sentiment: LIVE (NewsAPI)")
except Exception:
    st.sidebar.warning("⚠️ Sentiment: offline — add NEWS_API_KEY to Secrets")
    news_api_key = None

st.sidebar.success("✅ Neural Network: ONLINE")
st.sidebar.markdown("---")

st.sidebar.subheader("🎚️ Signal Thresholds")
buy_threshold  = st.sidebar.slider("BUY above (%)",  50, 80, 55)
sell_threshold = st.sidebar.slider("SELL below (%)", 20, 50, 45)
st.sidebar.caption(
    f"HOLD zone: {sell_threshold}% – {buy_threshold}%  |  "
    f"Net spread: {buy_threshold - sell_threshold}%"
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  11. UI — CONTROLS + MAIN ENGINE                             ║
# ╚══════════════════════════════════════════════════════════════╝
col_ctrl, col_main = st.columns([1, 2.5])

with col_ctrl:
    st.subheader("Laboratory Controls")
    selected_assets = st.multiselect(
        "Select Assets for Comparison:",
        AVAILABLE_STOCKS,
        default=["RELIANCE.NS"]
    )
    period = st.select_slider(
        "Select Data Timeframe",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="1y"
    )
    analyze_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

with col_main:
    if not analyze_btn:
        st.info("👈 Configure your parameters and hit **Run Simulation** to begin.")

    elif not user_api_key:
        st.error("🚨 API Key missing in Streamlit Secrets. Cannot proceed.")

    elif not selected_assets:
        st.warning("⚠️ Please select at least one stock.")

    else:
        # ── Fetch macro data once for all tickers ──────────────────
        with st.spinner("Fetching macro data (VIX · Oil · USD/INR)..."):
            macro_df = fetch_macro_data(period)

        if macro_df is None:
            st.error("❌ Macro data unavailable. Check your connection and retry.")
            st.stop()

        # ── Three tabs ─────────────────────────────────────────────
        tab_analysis, tab_backtest, tab_compare = st.tabs(
            ["📊 Live Analysis", "📉 Backtest", "🔗 Correlation"]
        )

        # ── Pre-build merged DataFrames once per ticker ────────────
        ticker_data: dict[str, pd.DataFrame] = {}
        for ticker in selected_assets:
            price_df = fetch_stock_data(ticker, period)
            if price_df is None:
                st.warning(f"⚠️ No price data returned for {ticker}. Skipping.")
                continue
            merged = price_df.join(macro_df, how='inner').ffill()
            merged['sentiment'] = 0.0
            merged = add_indicators(merged)
            if len(merged) >= SEQ_LEN:
                ticker_data[ticker] = merged

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 1 — LIVE ANALYSIS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with tab_analysis:
            for ticker, df in ticker_data.items():
                with st.spinner(f"Running AI on {ticker}..."):
                    try:
                        symbol = ticker.split('.')[0]

                        # — Fetch & inject live sentiment into prediction window —
                        sentiment_score, sentiment_label = (
                            get_sentiment_score(symbol, news_api_key)
                            if news_api_key else (0.0, "⚪ Neutral")
                        )
                        df = df.copy()
                        # Historical rows stay 0.0; only live window gets real score
                        df.iloc[-SEQ_LEN:, df.columns.get_loc('sentiment')] = sentiment_score

                        prob = run_prediction(df)
                        if prob is None:
                            st.warning(f"⚠️ Prediction failed for {ticker}.")
                            continue

                        target_price  = fetch_analyst_target(symbol, user_api_key)
                        current_price = df['close'].iloc[-1]
                        last_rsi      = df['rsi_14'].iloc[-1]

                        # — Metrics row —
                        st.subheader(f"📊 {ticker}")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Current Price",    f"₹ {current_price:.2f}")
                        m2.metric("Analyst Target",   f"₹ {target_price}")
                        m3.metric("AI Uptrend Prob.", f"{prob:.1f}%")
                        m4.metric("RSI (14)",         f"{last_rsi:.1f}")
                        m5.metric("News Sentiment",   sentiment_label,
                                  delta=f"{sentiment_score:+.3f}")

                        # — Signal banner —
                        if prob > buy_threshold:
                            st.success(
                                f"🤖 AI SIGNAL: **STRONG BUY** 🟢  "
                                f"(prob {prob:.1f}% > {buy_threshold}%)"
                            )
                        elif prob < sell_threshold:
                            st.error(
                                f"🤖 AI SIGNAL: **SELL / SHORT** 🔴  "
                                f"(prob {prob:.1f}% < {sell_threshold}%)"
                            )
                        else:
                            st.warning(
                                f"🤖 AI SIGNAL: **HOLD / NEUTRAL** ⚪  "
                                f"(prob {prob:.1f}% in hold zone)"
                            )

                        # — Technical chart —
                        st.plotly_chart(
                            chart_technical(df, ticker),
                            use_container_width=True
                        )
                        st.divider()

                    except Exception as e:
                        st.error(f"❌ {ticker} analysis error: {e}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 2 — BACKTEST
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with tab_backtest:
            st.caption(
                "Simulates running this model historically. "
                "BUY signal → holds long. SELL signal → holds short. "
                "HOLD signal → stays flat. No transaction costs included."
            )
            for ticker, df in ticker_data.items():
                with st.spinner(f"Backtesting {ticker}..."):
                    try:
                        bt = run_backtest(df, buy_threshold, sell_threshold)
                        if bt is None:
                            st.warning(f"⚠️ Not enough history to backtest {ticker}.")
                            continue

                        st.subheader(f"📉 {ticker}")

                        total_ret = bt['cum_strategy'].iloc[-1] - 1
                        bh_ret    = bt['cum_buyhold'].iloc[-1] - 1
                        n_buy     = (bt['signal'] == 'BUY').sum()
                        n_sell    = (bt['signal'] == 'SELL').sum()
                        n_hold    = (bt['signal'] == 'HOLD').sum()
                        alpha     = total_ret - bh_ret

                        b1, b2, b3, b4, b5 = st.columns(5)
                        b1.metric("AI Strategy Return",  f"{total_ret * 100:.1f}%")
                        b2.metric("Buy & Hold Return",   f"{bh_ret * 100:.1f}%",
                                  delta=f"Alpha: {alpha * 100:+.1f}%")
                        b3.metric("BUY Signals",  str(n_buy))
                        b4.metric("SELL Signals", str(n_sell))
                        b5.metric("HOLD Days",    str(n_hold))

                        st.plotly_chart(
                            chart_backtest(bt, ticker),
                            use_container_width=True
                        )
                        st.divider()

                    except Exception as e:
                        st.error(f"❌ Backtest error for {ticker}: {e}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 3 — CORRELATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with tab_compare:
            if len(selected_assets) < 2:
                st.info("ℹ️ Select **2 or more stocks** to see the correlation heatmap.")
            else:
                with st.spinner("Computing return correlations..."):
                    corr_fig = chart_correlation(list(ticker_data.keys()), period)

                if corr_fig:
                    st.subheader("🔗 Return Correlation Matrix")
                    st.caption(
                        "**+1.0** = move in perfect lockstep  |  "
                        "**0.0** = uncorrelated  |  "
                        "**−1.0** = move in opposite directions.  "
                        "Low correlation between holdings = better diversification."
                    )
                    st.plotly_chart(corr_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Not enough overlapping data to compute correlations.")