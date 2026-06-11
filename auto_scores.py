"""
auto_scores.py
==============
Automatically computes all research scores from live data.

Sources:
  - yfinance  : analyst consensus, margins, beta, price targets, D/E
  - Finnhub   : 3Y/5Y revenue & EPS growth, detailed financials, recommendation trends

Scores returned (all 0-10):
  forever_score : overall long-term conviction
  div_safety    : sustainability of options-premium dividend (Roundhill 2x ETFs)
  growth_lt     : 5-10 year growth runway
  moat          : competitive advantage (margins + ROE)
  risk          : safety / low volatility (10 = lowest risk)
  verdict       : auto-generated from forever_score

Cached for 24 hours — fundamentals don't move daily.
"""

import os
import numpy as np
import yfinance as yf
import streamlit as st
import finnhub as _finnhub


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo=0.0, hi=10.0) -> float:
    return round(max(lo, min(hi, v if v is not None and not np.isnan(v) else 5.0)), 2)


def _finnhub_key() -> str:
    # 1. Environment variable (Railway, CI, etc.)
    key = os.getenv("FINNHUB_API_KEY", "")
    if key:
        return key
    # 2. Read secrets.toml directly — works inside AND outside Streamlit runtime
    try:
        import tomllib
        _path = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
        with open(_path, "rb") as f:
            key = tomllib.load(f).get("FINNHUB_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 3. st.secrets fallback
    try:
        return st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        return ""


def _finnhub_client():
    key = _finnhub_key()
    return _finnhub.Client(api_key=key) if key else None


# ── Sub-scorers ───────────────────────────────────────────────────────────────

def _analyst_score(rec_trends: list) -> float:
    """
    Score from Finnhub recommendation_trends.
    strongBuy=+2, buy=+1, hold=0, sell=-1, strongSell=-2
    Mapped to 0-10.
    """
    if not rec_trends:
        return 5.0
    latest = rec_trends[0]
    sb  = latest.get("strongBuy",  0) or 0
    b   = latest.get("buy",        0) or 0
    h   = latest.get("hold",       0) or 0
    s   = latest.get("sell",       0) or 0
    ss  = latest.get("strongSell", 0) or 0
    total = sb + b + h + s + ss
    if total == 0:
        return 5.0
    weighted = (sb * 2 + b * 1 + h * 0 + s * -1 + ss * -2) / total
    # weighted ranges from -2 to +2; map to 0-10
    return _clamp((weighted + 2) / 4 * 10)


def _growth_score(pct: float | None) -> float:
    """Convert a growth % (e.g. 85.2 for 85%) to a 0-10 score."""
    if pct is None or np.isnan(pct):
        return 5.0
    if pct >= 150: return 10.0
    if pct >= 100: return 9.5
    if pct >=  60: return 9.0
    if pct >=  30: return 8.0
    if pct >=  20: return 7.0
    if pct >=  10: return 6.0
    if pct >=   5: return 5.0
    if pct >=   0: return 3.5
    if pct >= -10: return 2.0
    return 1.0


def _margin_score(pct: float | None) -> float:
    """Convert a margin % (e.g. 74.1 for 74%) to a 0-10 score.
    Raised thresholds — NVDA/MSFT/AVGO all exceed 70% so need more room at top."""
    if pct is None or np.isnan(pct):
        return 5.0
    if pct >= 85: return 10.0   # only true software/IP businesses (was 70)
    if pct >= 75: return  9.0
    if pct >= 65: return  8.0
    if pct >= 55: return  7.0
    if pct >= 40: return  6.0
    if pct >= 25: return  5.0
    if pct >= 15: return  3.5
    if pct >=  0: return  2.0
    return 1.0


def _roe_score(pct: float | None) -> float:
    """Convert ROE % to a 0-10 score."""
    if pct is None or np.isnan(pct):
        return 5.0
    if pct >= 80: return 10.0
    if pct >= 50: return  9.0
    if pct >= 30: return  8.0
    if pct >= 20: return  7.0
    if pct >= 15: return  6.0
    if pct >= 10: return  5.0
    if pct >=  5: return  4.0
    if pct >=  0: return  2.5
    return 1.0


def _beta_score(beta: float | None) -> float:
    """Lower beta → higher score (10 = least volatile)."""
    if beta is None or np.isnan(beta):
        return 5.0
    if beta <= 0.5: return 10.0
    if beta <= 0.8: return  8.5
    if beta <= 1.0: return  7.5
    if beta <= 1.3: return  6.5
    if beta <= 1.5: return  5.5
    if beta <= 1.8: return  4.5
    if beta <= 2.0: return  3.5
    if beta <= 2.5: return  2.5
    return 1.5


def _de_score(de: float | None) -> float:
    """Lower debt/equity → higher score (10 = no debt)."""
    if de is None or np.isnan(de):
        return 5.0
    if de <= 0.0:  return 10.0
    if de <= 0.1:  return  9.5
    if de <= 0.3:  return  9.0
    if de <= 0.5:  return  8.0
    if de <= 1.0:  return  7.0
    if de <= 2.0:  return  5.5
    if de <= 3.0:  return  4.0
    if de <= 5.0:  return  2.5
    return 1.5


def _upside_score(target: float | None, current: float | None) -> float:
    """Analyst price target upside % → 0-10 score."""
    if not target or not current or current <= 0:
        return 5.0
    upside = (target / current - 1) * 100
    if upside >= 50: return 10.0
    if upside >= 30: return  8.5
    if upside >= 20: return  7.5
    if upside >= 10: return  6.5
    if upside >=  0: return  5.0
    if upside >= -10: return 3.5
    return 2.0


def _auto_verdict(forever: float) -> str:
    if forever >= 8.5: return "STRONG BUY"
    if forever >= 7.0: return "BUY"
    if forever >= 5.5: return "HOLD"
    if forever >= 4.0: return "CAUTION"
    return "AVOID"


# ── Main scoring function ─────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)   # cache 24 hours
def compute_scores(ticker: str) -> dict:
    """
    Returns a dict with auto-computed scores for the given stock ticker.
    Falls back to neutral (5.0) for any metric that can't be fetched.
    """

    # ── yfinance ─────────────────────────────────────────────────────────────
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    rec_mean      = info.get("recommendationMean")     # 1=strong buy … 5=strong sell
    n_analysts    = info.get("numberOfAnalystOpinions", 0) or 0
    beta          = info.get("beta")
    target_price  = info.get("targetMeanPrice")
    curr_price    = info.get("currentPrice") or info.get("regularMarketPrice")

    # ── Finnhub ───────────────────────────────────────────────────────────────
    client = _finnhub_client()
    rec_trends, metrics = [], {}
    if client:
        try:
            rec_trends = client.recommendation_trends(ticker) or []
        except Exception:
            pass
        try:
            fin     = client.company_basic_financials(ticker, "all")
            metrics = fin.get("metric", {}) or {}
        except Exception:
            pass

    # Pull Finnhub metrics (all in % already, e.g. 100.05 = 100%)
    rev3y    = metrics.get("revenueGrowth3Y")
    rev5y    = metrics.get("revenueGrowth5Y")
    eps3y    = metrics.get("epsGrowth3Y")
    eps5y    = metrics.get("epsGrowth5Y")
    gross_m  = metrics.get("grossMarginTTM")     # e.g. 74.15 = 74%
    _op_yf   = info.get("operatingMargins")      # yfinance decimal e.g. 0.656
    op_m     = metrics.get("operatingMarginTTM") or (_op_yf * 100 if _op_yf else None)
    roe      = metrics.get("roeTTM")             # e.g. 111.66 = 112%
    de_ratio = metrics.get("totalDebt/totalEquityAnnual")  # already a ratio

    # ── Sub-scores ────────────────────────────────────────────────────────────
    analyst_sc  = _analyst_score(rec_trends)
    rev3y_sc    = _growth_score(rev3y)
    rev5y_sc    = _growth_score(rev5y)
    eps3y_sc    = _growth_score(eps3y)
    eps5y_sc    = _growth_score(eps5y)
    gross_sc    = _margin_score(gross_m)
    op_sc       = _margin_score(op_m)
    roe_sc      = _roe_score(roe)
    beta_sc     = _beta_score(beta)
    de_sc       = _de_score(de_ratio)
    upside_sc   = _upside_score(target_price, curr_price)

    # ── Composite scores ──────────────────────────────────────────────────────

    # Forever Score — overall long-term conviction
    # Reduced analyst weight (all big tech = strong buy → inflates score)
    # Upside raised — price target gap is a real differentiator
    forever = _clamp(
        eps3y_sc    * 0.30 +   # earnings growth most predictive of LT returns
        rev3y_sc    * 0.25 +   # revenue growth shows market expansion
        analyst_sc  * 0.20 +   # consensus still matters, but less dominant
        upside_sc   * 0.15 +   # price target gap captures valuation buffer
        gross_sc    * 0.10     # margin quality as tiebreaker
    )

    # Div Safety — sustainability of weekly options-premium distributions
    # Uses D/E instead of ROE — debt burden directly threatens distributions
    # Gross margin reflects cash generation reliability of the underlying
    div_safety = _clamp(
        gross_sc    * 0.40 +   # high margins = reliable underlying cash = stable premium
        analyst_sc  * 0.30 +   # consensus confidence in underlying direction
        de_sc       * 0.30     # low debt = distributions not threatened by balance sheet
    )

    # Growth LT — 5-10 year runway
    growth_lt = _clamp(
        eps5y_sc    * 0.35 +
        rev5y_sc    * 0.35 +
        upside_sc   * 0.20 +
        analyst_sc  * 0.10
    )

    # Moat — competitive advantage (margins + returns + balance sheet quality)
    moat = _clamp(
        gross_sc    * 0.40 +   # gross margin is the purest moat signal
        op_sc       * 0.30 +   # operating margin shows execution discipline
        roe_sc      * 0.20 +   # returns on equity show capital efficiency
        de_sc       * 0.10     # low debt amplifies the moat (financial flexibility)
    )

    # Risk — 10 = lowest risk
    # Restructured: D/E dominates (company survival for forever-hold)
    # Beta only 25% — high beta underlyings generate MORE option premium income
    risk = _clamp(
        de_sc       * 0.75 +
        beta_sc     * 0.25
    )

    verdict = _auto_verdict(forever)

    return {
        "forever_score": forever,
        "div_safety":    div_safety,
        "growth_lt":     growth_lt,
        "moat":          moat,
        "risk":          risk,
        "verdict":       verdict,
        # Raw sub-scores for display
        "_analyst":      round(analyst_sc,  1),
        "_rev3y":        round(rev3y_sc,    1),
        "_eps3y":        round(eps3y_sc,    1),
        "_gross_margin": round(gross_sc,    1),
        "_op_margin":    round(op_sc,       1),
        "_roe":          round(roe_sc,       1),
        "_beta":         round(beta_sc,     1),
        "_upside":       round(upside_sc,   1),
        # Raw values for tooltips
        "_raw_rev3y":    rev3y,
        "_raw_eps3y":    eps3y,
        "_raw_rev5y":    rev5y,
        "_raw_eps5y":    eps5y,
        "_raw_gross":    gross_m,
        "_raw_op":       op_m,
        "_raw_roe":      roe,
        "_raw_beta":     beta,
        "_raw_de":       de_ratio,
        "_raw_upside":   round((target_price / curr_price - 1) * 100, 1)
                         if target_price and curr_price and curr_price > 0 else None,
        "_n_analysts":   n_analysts,
    }


def get_scores(ticker: str) -> dict:
    """Public entry point — returns compute_scores with safe fallback."""
    try:
        return compute_scores(ticker)
    except Exception:
        return {
            "forever_score": 5.0, "div_safety": 5.0,
            "growth_lt": 5.0, "moat": 5.0, "risk": 5.0,
            "verdict": "HOLD",
        }


def unified_score(
    ticker: str,
    nav_score:      float | None = None,
    yield_score:    float | None = None,
    momentum_score: float | None = None,
) -> float:
    """
    Unified score (0-10) for Roundhill WeeklyPay ETF selection.

    Rationale — WeeklyPay ETFs are 1.2x leveraged swaps on the underlying stock.
    As the underlying grows, the ETF grows 1.2x and weekly distributions grow
    proportionally. Income is NOT from the underlying's dividend — it comes from
    the option premium generated by the swap structure. Therefore:

    Fundamental components (underlying stock quality):
      Growth LT   28% — stock grows → ETF grows 1.2x → distributions grow (primary driver)
      Moat        20% — strong competitive advantage sustains the compounding
      Forever     14% — overall conviction: EPS growth, analyst consensus, earnings quality
      Risk         7% — D/E focused (company survival), NOT beta
                         (high beta = more option premium = more income)

    ETF instrument components (what the actual ETF is doing right now):
      Momentum    18% — underlying trending above MAs = 1.2x swap working positively
      Yield       10% — current weekly income rate from the ETF
      NAV          3% — capital preservation sanity check

    Total: 100% — no separate blend step needed.

    When ETF components are not available (research tab before screener visited),
    falls back to 5.0 (neutral) for each missing ETF component.

    Verdict thresholds:
      ≥ 8.0  → STRONG BUY
      ≥ 7.0  → BUY
      ≥ 5.5  → HOLD
      ≥ 4.0  → CAUTION
      < 4.0  → AVOID
    """
    s = get_scores(ticker)

    # ETF instrument scores — use neutral 5.0 if not yet available
    nav_sc  = float(nav_score)      if nav_score      is not None else 5.0
    yld_sc  = float(yield_score)    if yield_score    is not None else 5.0
    mom_sc  = float(momentum_score) if momentum_score is not None else 5.0

    score = (
        s.get("growth_lt",     5.0) * 0.28 +
        s.get("moat",          5.0) * 0.20 +
        mom_sc                      * 0.18 +
        s.get("forever_score", 5.0) * 0.14 +
        yld_sc                      * 0.10 +
        s.get("risk",          5.0) * 0.07 +
        nav_sc                      * 0.03
    )

    return round(max(0.0, min(10.0, score)), 2)


def unified_verdict(score: float) -> str:
    if score >= 8.0: return "STRONG BUY"
    if score >= 7.0: return "BUY"
    if score >= 5.5: return "HOLD"
    if score >= 4.0: return "CAUTION"
    return "AVOID"
