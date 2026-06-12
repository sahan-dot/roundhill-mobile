"""
Roundhill Mobile Dashboard
Lightweight Streamlit app for on-the-go portfolio monitoring.
Deployed on Streamlit Community Cloud.
"""

import json
import os
import urllib.request
from datetime import datetime, date, timedelta

import finnhub as _finnhub
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from auto_scores import unified_score, unified_verdict, get_scores
from roundhill_scraper import scrape_multiple_roundhill_etfs, is_roundhill_etf

# ── Config ───────────────────────────────────────────────────────────────────

SNAPSHOT_API_URL = "https://api.github.com/repos/sahan-dot/roundhill-bot/contents/portfolio_snapshot.json"
SCORE_HISTORY_API_URL = "https://api.github.com/repos/sahan-dot/roundhill-bot/contents/score_history.json"

ETF_MAP = {
    "AAPW": {"underlying": "AAPL",  "name": "Apple"},
    "AMDW": {"underlying": "AMD",   "name": "AMD"},
    "AMZW": {"underlying": "AMZN",  "name": "Amazon"},
    "ARMW": {"underlying": "ARM",   "name": "ARM Holdings"},
    "AVGW": {"underlying": "AVGO",  "name": "Broadcom"},
    "BRKW": {"underlying": "BRK-B", "name": "Berkshire"},
    "COSW": {"underlying": "COST",  "name": "Costco"},
    "GOOW": {"underlying": "GOOGL", "name": "Alphabet"},
    "HOOW": {"underlying": "HOOD",  "name": "Robinhood"},
    "METW": {"underlying": "META",  "name": "Meta"},
    "MSFW": {"underlying": "MSFT",  "name": "Microsoft"},
    "NVDW": {"underlying": "NVDA",  "name": "NVIDIA"},
    "PLTW": {"underlying": "PLTR",  "name": "Palantir"},
    "TSLW": {"underlying": "TSLA",  "name": "Tesla"},
    "UBEW": {"underlying": "UBER",  "name": "Uber"},
    "UNHW": {"underlying": "UNH",   "name": "UnitedHealth"},
}

WHT_RATE = 0.15

START = (date.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

NEGATIVE_WORDS = {
    "lawsuit","fraud","investigation","subpoena","recall","downgrade",
    "miss","missed","loss","losses","decline","fell","fall","cut","cuts",
    "layoff","breach","fine","penalty","warning","delay","default",
    "bankruptcy","probe","charges","violation","disappoints","weak",
    "sell","underperform","drops","slumps","plunges","sinks",
}
POSITIVE_WORDS = {
    "beat","beats","record","upgrade","raise","raised","growth","surges",
    "jumps","soars","outperform","buy","strong","partnership","deal",
    "contract","profit","approval","approved","launch","breakthrough",
    "milestone","positive","gains","rallies","climbs",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot():
    try:
        gh_token = st.secrets.get("GITHUB_TOKEN", "")
        req = urllib.request.Request(SNAPSHOT_API_URL)
        req.add_header("Accept", "application/vnd.github.v3.raw")
        if gh_token:
            req.add_header("Authorization", f"Bearer {gh_token}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        st.error(f"Debug: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_score_history():
    """Load dated score history from GitHub. Returns {date_str: {etf: {score, verdict}}}."""
    try:
        gh_token = st.secrets.get("GITHUB_TOKEN", "")
        req = urllib.request.Request(SCORE_HISTORY_API_URL)
        req.add_header("Accept", "application/vnd.github.v3.raw")
        if gh_token:
            req.add_header("Authorization", f"Bearer {gh_token}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if "history" in data:
            return data["history"]
        # Legacy flat format — treat as yesterday
        return {(date.today() - timedelta(days=1)).isoformat(): data}
    except Exception:
        return {}


def _yf_price(symbol: str) -> float | None:
    try:
        tk = yf.Ticker(symbol)
        p = getattr(tk.fast_info, "last_price", None)
        if p and p > 0:
            return float(p)
        hist = tk.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _finnhub_client():
    key = st.secrets.get("FINNHUB_API_KEY", "") if hasattr(st, "secrets") else ""
    if not key:
        key = os.getenv("FINNHUB_API_KEY", "")
    return _finnhub.Client(api_key=key) if key else None


def _verdict_emoji(v):
    return {"STRONG BUY": "🟢", "BUY": "🔵", "HOLD": "🟡",
            "CAUTION": "🟠", "AVOID": "🔴"}.get(v, "⚪")


def _flag(title: str) -> str:
    tl = title.lower()
    if any(w in tl for w in NEGATIVE_WORDS):
        return "🔴"
    if any(w in tl for w in POSITIVE_WORDS):
        return "🟢"
    return "⚪"


# ── ETF-level scoring (ported from roundhill_web.py) ─────────────────────────

def _strip_tz(idx):
    if hasattr(idx, "tz") and idx.tz is not None:
        return idx.tz_localize(None)
    return idx


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_price(ticker):
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df if len(df) > 10 else None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_dividends(ticker):
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs is None or len(divs) == 0:
            return pd.Series(dtype=float)
        divs.index = _strip_tz(divs.index)
        return divs[divs.index >= pd.Timestamp(START)]
    except Exception:
        return pd.Series(dtype=float)


def _nav_health(price_df, divs):
    if price_df is None or len(price_df) < 5:
        return None
    fp = float(price_df["Close"].iloc[0])
    lp = float(price_df["Close"].iloc[-1])
    pch = (lp - fp) / fp * 100
    age_days = (price_df.index[-1] - price_df.index[0]).days
    yrs = max(age_days / 365.25, 0.05)
    ann = pch / yrs
    tdp = float(divs.sum()) / fp * 100 if len(divs) > 0 else 0.0
    tr = pch + tdp
    if   ann >= 50:  ns = 10.0
    elif ann >= 30:  ns = 9.0
    elif ann >= 15:  ns = 8.0
    elif ann >= 8:   ns = 7.0
    elif ann >= 0:   ns = 6.0
    elif ann >= -5:  ns = 4.5
    elif ann >= -10: ns = 3.0
    elif ann >= -20: ns = 1.5
    elif ann >= -35: ns = 0.5
    else:            ns = 0.0
    if tr > 0 and pch < 0:
        ns = min(ns + 0.5, 10.0)
    return {"nav_score": ns}


def _detect_div_freq(divs):
    if len(divs) < 2:
        return 52, 4, 12
    gaps = divs.sort_index().index.to_series().diff().dropna().dt.days
    avg = gaps.mean()
    if avg <= 14:  return 52, 4, 12
    if avg <= 45:  return 12, 3, 9
    if avg <= 100: return 4, 2, 6
    return 1, 1, 2


def _yield_analysis(price_df, divs):
    if len(divs) < 2:
        return {"yield_score": 0}
    cp = float(price_df["Close"].iloc[-1])
    ds = divs.sort_index()
    freq, recent_n, lookback = _detect_div_freq(ds)
    r_recent = ds.iloc[-recent_n:]
    old = ds.iloc[-lookback:-recent_n] if len(ds) >= lookback else ds.iloc[:-recent_n]
    aw = float(r_recent.mean())
    ao = float(old.mean()) if len(old) > 0 else aw
    ay = aw * freq / cp * 100
    trend = "N/A"
    if len(ds) >= recent_n * 2:
        trend = ("Rising" if aw > ao * 1.05 else
                 "Falling" if aw < ao * 0.95 else "Stable")
    if   ay >= 100: ys = 10.0
    elif ay >= 80:  ys = 9.0
    elif ay >= 60:  ys = 8.0
    elif ay >= 45:  ys = 7.0
    elif ay >= 30:  ys = 6.0
    elif ay >= 20:  ys = 5.0
    elif ay >= 12:  ys = 3.5
    elif ay >= 6:   ys = 2.0
    else:           ys = 0.5
    if trend == "Falling": ys = max(ys - 1.5, 0)
    elif trend == "Rising": ys = min(ys + 0.5, 10.0)
    return {"yield_score": ys}


def _underlying_analysis(und):
    df = _fetch_price(und)
    if df is None or len(df) < 50:
        return None
    c = float(df["Close"].iloc[-1])
    def sr(n): return (c / float(df["Close"].iloc[-n]) - 1) * 100 if len(df) > n else np.nan
    r1m, r3m, r6m = sr(21), sr(63), sr(126)
    ma50 = float(df["Close"].iloc[-50:].mean()) if len(df) >= 50 else np.nan
    ma200 = float(df["Close"].iloc[-200:].mean()) if len(df) >= 200 else np.nan
    a50 = (c > ma50) if not np.isnan(ma50) else None
    a200 = (c > ma200) if not np.isnan(ma200) else None
    sc = 5.0
    if a200:  sc += 1.5
    if a50:   sc += 1.0
    if not np.isnan(r1m):
        if   r1m > 10: sc += 0.75
        elif r1m > 3:  sc += 0.5
        elif r1m > 0:  sc += 0.25
    if not np.isnan(r3m):
        if   r3m > 20: sc += 1.0
        elif r3m > 8:  sc += 0.75
        elif r3m > 0:  sc += 0.35
    if not np.isnan(r6m):
        if   r6m > 30: sc += 1.0
        elif r6m > 12: sc += 0.75
        elif r6m > 0:  sc += 0.35
    if not a200: sc -= 2.0
    if not a50:  sc -= 1.0
    if not np.isnan(r3m) and r3m < -15: sc -= 1.0
    if not np.isnan(r6m) and r6m < -20: sc -= 1.5
    sc = max(0.0, min(10.0, sc))
    return {"score": sc}


def _compute_etf_scores(etf_ticker, underlying):
    """Compute nav_score, yield_score, momentum_score for an ETF."""
    nav_sc, yld_sc, mom_sc = None, None, None
    try:
        price_df = _fetch_price(etf_ticker)
        divs = _fetch_dividends(etf_ticker)
        if price_df is not None:
            nav = _nav_health(price_df, divs)
            if nav:
                nav_sc = nav["nav_score"]
            ya = _yield_analysis(price_df, divs)
            if ya:
                yld_sc = ya["yield_score"]
        ua = _underlying_analysis(underlying)
        if ua:
            mom_sc = ua["score"]
    except Exception:
        pass
    return nav_sc, yld_sc, mom_sc


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Roundhill Mobile", page_icon="📊", layout="centered")
st.title("📊 Roundhill Dashboard")

snap = load_snapshot()
if not snap:
    st.error("Could not load portfolio snapshot from GitHub.")
    st.stop()

positions = snap.get("positions", [])
pos_syms = [p["symbol"] for p in positions]
qty_map = {p["symbol"]: p["quantity"] for p in positions}

tab1, tab2, tab3, tab4 = st.tabs(["📈 MTD Returns", "⭐ Scores", "📰 News", "💸 Dividends"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MTD Returns
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Use pre-computed MTD figures from the main Streamlit app's snapshot
    mtd_pct = snap.get("mtd_pct", 0)
    mtd_total = snap.get("mtd_total", 0)
    mtd_price_pl = snap.get("mtd_price_pl", 0)
    mtd_div = snap.get("mtd_dividends", 0)
    mtd_realized = snap.get("mtd_realized", 0)
    month_start_val = snap.get("month_start_value_aud", 0)
    cash_aud = snap.get("cash_aud", 0)
    snap_port_val = snap.get("portfolio_value", 0)

    # Live portfolio value for display
    with st.spinner("Fetching live prices..."):
        audusd = _yf_price("AUDUSD=X")
        prices = {}
        for p in positions:
            prices[p["symbol"]] = _yf_price(p["symbol"])

    if audusd and all(prices.get(s) for s in pos_syms):
        equity_usd = sum(qty_map[s] * prices[s] for s in pos_syms)
        equity_aud = equity_usd / audusd
        port_val = equity_aud + cash_aud
    else:
        port_val = snap_port_val

    c1, c2 = st.columns(2)
    c1.metric("Portfolio Value", f"A${port_val:,.0f}")
    c2.metric("MTD Return", f"{mtd_pct:+.2f}%", delta=f"A${mtd_total:+,.0f}")

    st.caption(f"Price P&L: A${mtd_price_pl:+,.0f}  |  Dividends: A${mtd_div:+,.0f}  |  Realized: A${mtd_realized:+,.0f}")

    # Cumulative MTD chart
    today = date.today()
    mtd_start = today.replace(day=1)
    try:
        price_data = yf.download(
            pos_syms + ["AUDUSD=X"],
            start=(mtd_start - timedelta(days=5)).strftime("%Y-%m-%d"),
            auto_adjust=False, progress=False,
        )["Close"]

        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = price_data.columns.get_level_values(-1)

        price_data = price_data.dropna(how="all")
        if not price_data.empty and "AUDUSD=X" in price_data.columns:
            fx_col = price_data["AUDUSD=X"].ffill()
            daily_equity = pd.Series(0.0, index=price_data.index)
            for sym in pos_syms:
                if sym in price_data.columns:
                    daily_equity += price_data[sym].ffill() * qty_map[sym]
            daily_equity_aud = daily_equity / fx_col

            pre_month = daily_equity_aud[daily_equity_aud.index < pd.Timestamp(mtd_start)]
            base = pre_month.iloc[-1] if not pre_month.empty else daily_equity_aud.iloc[0]

            month_data = daily_equity_aud[daily_equity_aud.index >= pd.Timestamp(mtd_start)]
            if not month_data.empty:
                cum_ret = ((month_data - base) / month_start_val * 100).round(2)
                chart_df = pd.DataFrame({"MTD Return %": cum_ret.values}, index=cum_ret.index)
                st.line_chart(chart_df, y="MTD Return %", use_container_width=True)
    except Exception:
        pass

    st.caption(f"Snapshot updated: {snap.get('updated', '?')}  |  MTD from Streamlit app")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Claude Scores (with full ETF-level scoring)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("⭐ Claude Scores")

    portfolio_etfs = [s for s in pos_syms if s in ETF_MAP]
    all_etfs = portfolio_etfs + [e for e in ETF_MAP if e not in portfolio_etfs]

    # Load previous scores for delta calculation
    score_hist = load_score_history()
    sorted_dates = sorted(score_hist.keys())
    # Find yesterday's scores (most recent date that isn't today)
    today_str = date.today().isoformat()
    prev_scores = {}
    for d in reversed(sorted_dates):
        if d != today_str:
            prev_scores = score_hist[d]
            prev_date = d
            break

    with st.spinner("Computing scores..."):
        rows = []
        for etf in all_etfs:
            meta = ETF_MAP.get(etf, {})
            underlying = meta.get("underlying", etf)
            try:
                nav_sc, yld_sc, mom_sc = _compute_etf_scores(etf, underlying)
                score = unified_score(underlying, nav_score=nav_sc, yield_score=yld_sc, momentum_score=mom_sc)
                verdict = unified_verdict(score)
            except Exception:
                score = None
                verdict = "N/A"

            # Calculate delta from previous day
            delta_str = ""
            if score is not None and etf in prev_scores:
                prev = prev_scores[etf]
                prev_val = prev.get("claude_score") or prev.get("score")
                if prev_val is not None:
                    delta = score - float(prev_val)
                    if abs(delta) >= 0.05:
                        arrow = "▲" if delta > 0 else "▼"
                        delta_str = f" {arrow}{abs(delta):+.1f}"[:-0] if delta else ""
                        delta_str = f" {arrow}{abs(delta):.1f}"

            in_portfolio = "✅" if etf in portfolio_etfs else ""
            rows.append({
                "": in_portfolio,
                "ETF": etf,
                "Underlying": meta.get("name", underlying),
                "Score": (f"{score:.1f}{delta_str}" if score else "—"),
                "Verdict": f"{_verdict_emoji(verdict)} {verdict}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    if prev_scores:
        st.caption(f"Changes vs. {prev_date}")
    else:
        st.caption("Score changes will appear after the next daily alert run.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — News (with sentiment highlighting)
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("📰 Portfolio News")

    fh = _finnhub_client()
    if not fh:
        st.warning("Set FINNHUB_API_KEY in Streamlit secrets for news.")
    else:
        portfolio_underlyings = [(s, ETF_MAP[s]["underlying"], ETF_MAP[s]["name"])
                                 for s in pos_syms if s in ETF_MAP]

        # Market news
        with st.spinner("Fetching market news..."):
            try:
                mkt_news = fh.general_news("general", min_id=0)[:5]
                st.markdown("**🌍 Market News**")
                for item in mkt_news:
                    headline = item.get("headline", "")
                    url = item.get("url", "")
                    source = item.get("source", "")
                    ts = item.get("datetime", 0)
                    age = ""
                    if ts:
                        delta = datetime.now() - datetime.fromtimestamp(ts)
                        if delta.days > 0:
                            age = f"{delta.days}d ago"
                        else:
                            hrs = delta.seconds // 3600
                            age = f"{hrs}h ago" if hrs else "just now"
                    sentiment = _flag(headline)
                    st.markdown(f"- {sentiment} [{headline}]({url})  \n  *{source} · {age}*")
            except Exception as e:
                st.info(f"Could not fetch market news: {e}")

        st.divider()

        # Per-stock news
        for etf, underlying, name in portfolio_underlyings:
            try:
                news = fh.company_news(underlying,
                                       _from=(date.today() - timedelta(days=3)).isoformat(),
                                       to=date.today().isoformat())[:3]
                if news:
                    st.markdown(f"**{etf}** ({name})")
                    for item in news:
                        headline = item.get("headline", "")
                        url = item.get("url", "")
                        source = item.get("source", "")
                        sentiment = _flag(headline)
                        st.markdown(f"- {sentiment} [{headline}]({url})  \n  *{source}*")
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dividends (with new-data alert)
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("💸 Weekly Distributions")

    rh_syms = [s for s in pos_syms if is_roundhill_etf(s)]

    with st.spinner("Scraping Roundhill distributions..."):
        rh_data = scrape_multiple_roundhill_etfs(rh_syms)

    audusd_rate = _yf_price("AUDUSD=X") or 0.714
    usd_aud = 1 / audusd_rate

    rows = []
    total_gross = 0
    total_net = 0
    latest_ex = None
    latest_pay = None

    for sym in rh_syms:
        divs = rh_data.get(sym, [])
        if not divs:
            continue
        latest = divs[0]
        qty = qty_map.get(sym, 0)
        amount = latest["amount"]
        price = _yf_price(sym) or 0
        yield_pct = (amount / price * 100) if price > 0 else 0
        gross_usd = amount * qty
        gross_aud = gross_usd * usd_aud
        net_aud = gross_aud * (1 - WHT_RATE)

        total_gross += gross_aud
        total_net += net_aud

        if not latest_ex:
            latest_ex = latest.get("ex_date")
            latest_pay = latest.get("pay_date")

        rows.append({
            "ETF": sym,
            "Shares": int(qty),
            "$/Share": f"${amount:.6f}",
            "Yield %": f"{yield_pct:.2f}%",
            "Gross (A$)": f"${gross_aud:,.2f}",
            "Net (A$)": f"${net_aud:,.2f}",
        })

    if rows:
        # Check if dividend data is new since last visit
        if "last_seen_ex_date" not in st.session_state:
            st.session_state.last_seen_ex_date = None

        if latest_ex and latest_ex != st.session_state.last_seen_ex_date:
            if st.session_state.last_seen_ex_date is not None:
                st.success(f"🔔 New dividend data! Ex-date: **{latest_ex}** | Pay date: **{latest_pay}**")
                st.balloons()
            st.session_state.last_seen_ex_date = latest_ex

        st.caption(f"Ex-Date: **{latest_ex}**  |  Pay Date: **{latest_pay}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Gross", f"A${total_gross:,.2f}")
        c2.metric("WHT (15%)", f"-A${total_gross - total_net:,.2f}")
        c3.metric("Total Net", f"A${total_net:,.2f}")

        st.caption(f"USD/AUD: {usd_aud:.4f}")
    else:
        st.info("No distribution data available yet. Check back Friday ~10 AM AEST.")
