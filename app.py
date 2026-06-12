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


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot():
    try:
        gh_token = st.secrets.get("GITHUB_TOKEN", "")
        req = urllib.request.Request(SNAPSHOT_API_URL)
        req.add_header("Accept", "application/vnd.github.v3.raw")
        if gh_token:
            req.add_header("Authorization", f"token {gh_token}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


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
    month_start_val = snap.get("month_start_value_aud", 0)
    mtd_div = snap.get("mtd_dividends", 0)
    mtd_realized = snap.get("mtd_realized", 0)
    cash_aud = snap.get("cash_aud", 0)

    with st.spinner("Fetching live prices..."):
        audusd = _yf_price("AUDUSD=X")
        prices = {}
        for p in positions:
            prices[p["symbol"]] = _yf_price(p["symbol"])

    if audusd and all(prices.get(s) for s in pos_syms):
        equity_usd = sum(qty_map[s] * prices[s] for s in pos_syms)
        equity_aud = equity_usd / audusd
        port_val = equity_aud + cash_aud

        month_start_equity = month_start_val - cash_aud
        mtd_price = equity_aud - month_start_equity
        mtd_total = mtd_price + mtd_div + mtd_realized
        mtd_pct = (mtd_total / month_start_val * 100) if month_start_val else 0

        c1, c2 = st.columns(2)
        c1.metric("Portfolio Value", f"A${port_val:,.0f}")
        c2.metric("MTD Return", f"{mtd_pct:+.2f}%", delta=f"A${mtd_total:+,.0f}")

        st.caption(f"Price P&L: A${mtd_price:+,.0f}  |  Dividends: A${mtd_div:+,.0f}")

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

        st.caption(f"Snapshot updated: {snap.get('updated', '?')}  |  Prices: yfinance live")
    else:
        st.warning("Could not fetch all live prices.")
        mtd_pct = snap.get("mtd_pct", 0)
        mtd_total = snap.get("mtd_total", 0)
        st.metric("MTD Return (stored)", f"{mtd_pct:+.2f}%", delta=f"A${mtd_total:+,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Claude Scores
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("⭐ Claude Scores")

    portfolio_etfs = [s for s in pos_syms if s in ETF_MAP]
    all_etfs = portfolio_etfs + [e for e in ETF_MAP if e not in portfolio_etfs]

    with st.spinner("Computing scores..."):
        rows = []
        for etf in all_etfs:
            meta = ETF_MAP.get(etf, {})
            underlying = meta.get("underlying", etf)
            try:
                score = unified_score(underlying)
                verdict = unified_verdict(score)
            except Exception:
                score = None
                verdict = "N/A"

            in_portfolio = "✅" if etf in portfolio_etfs else ""
            rows.append({
                "": in_portfolio,
                "ETF": etf,
                "Underlying": meta.get("name", underlying),
                "Score": f"{score:.1f}" if score else "—",
                "Verdict": f"{_verdict_emoji(verdict)} {verdict}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — News
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
                    st.markdown(f"- [{headline}]({url})  \n  *{source} · {age}*")
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
                        st.markdown(f"- [{headline}]({url})  \n  *{source}*")
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dividends
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
