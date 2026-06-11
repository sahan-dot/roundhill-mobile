"""
Roundhill Dividend Scraper
Fetches distribution history directly from roundhillinvestments.com
via their internal PHP API (no Playwright / headless browser needed).

Flow:
  1. GET  /assets/php/server.php          → session token
  2. POST /assets/php/distribution-call.php  { upperetf, loweretf, token, is_ajax }
     → JSON array of [declaration, ex_date, record_date, pay_date, amount]

Data is typically available ~10 AM Brisbane (AEST) on Fridays.
"""

import json
import logging
import re
import requests
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL    = "https://www.roundhillinvestments.com"
TOKEN_URL   = f"{BASE_URL}/assets/php/server.php"
DISTRI_URL  = f"{BASE_URL}/assets/php/distribution-call.php"

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

# All supported Roundhill WeeklyPay / income ETFs
ETF_URL_MAPPING = {
    "MSTW": "mstw", "NVDW": "nvdw", "TSLW": "tslw", "AAPW": "aapw",
    "COIW": "coiw", "AMZW": "amzw", "GOOW": "goow", "METW": "metw",
    "MSFW": "msfw", "AMDW": "amdw", "AVGW": "avgw", "ARMW": "armw",
    "BABW": "babw", "BRKW": "brkw", "COSW": "cosw", "GDXW": "gdxw",
    "GLDW": "gldw", "HOOW": "hoow", "MAGY": "magy", "NFLW": "nflw",
    "PLTW": "pltw", "UBEW": "ubew", "UNHW": "unhw", "TSYW": "tsyw",
    "WEEK": "week", "TOPW": "topw", "YBTC": "ybtc", "YETH": "yeth",
    "QDTE": "qdte", "XDTE": "xdte", "RDTE": "rdte", "XPAY": "xpay",
    "WDTE": "wdte",
}


def _get_token(session: requests.Session) -> Optional[str]:
    """Fetch session token from Roundhill server.php."""
    try:
        r = session.get(TOKEN_URL, timeout=10)
        r.raise_for_status()
        token = r.text.strip()
        if not token:
            raise ValueError("Empty token returned")
        return token
    except Exception as e:
        logger.error(f"Failed to get Roundhill token: {e}")
        return None


def _parse_date(raw: str) -> Optional[str]:
    """Parse any date string from the API into YYYY-MM-DD."""
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def scrape_roundhill_dividends(ticker: str) -> tuple[list[dict], Optional[str]]:
    """
    Fetch full distribution history for one ETF.

    Returns:
        (dividends, error_message)
        dividends: list of dicts sorted newest-first:
            { declaration, ex_date, record_date, pay_date, amount }
        error_message: str if failed, else None
    """
    ticker = ticker.upper()
    if ticker not in ETF_URL_MAPPING:
        return [], f"Ticker {ticker} not in Roundhill ETF list"

    session = requests.Session()
    session.headers.update(_HEADERS)
    session.headers["Referer"] = f"{BASE_URL}/etf/{ETF_URL_MAPPING[ticker]}/"

    token = _get_token(session)
    if not token:
        return [], "Could not obtain session token from Roundhill"

    try:
        data = {
            "upperetf": ticker,
            "loweretf": ticker.lower() + "/",
            "token":    token,
            "is_ajax":  "1",
        }
        r = session.post(DISTRI_URL, data=data, timeout=15)
        r.raise_for_status()
        raw = json.loads(r.text)
    except json.JSONDecodeError:
        return [], f"Non-JSON response from Roundhill for {ticker}"
    except Exception as e:
        return [], f"Request failed for {ticker}: {e}"

    dividends = []
    # raw is a list (or dict of index→row) of [declaration, ex_date, record_date, pay_date, amount]
    rows = raw.values() if isinstance(raw, dict) else raw
    for row in rows:
        try:
            if row[0] == "Declaration":      # skip header row if present
                continue
            amount = float(row[4])
            if amount <= 0:
                continue
            dividends.append({
                "declaration": _parse_date(row[0]),
                "ex_date":     _parse_date(row[1]),
                "record_date": _parse_date(row[2]),
                "pay_date":    _parse_date(row[3]),
                "amount":      round(amount, 6),
            })
        except (IndexError, ValueError, TypeError):
            continue

    dividends.sort(key=lambda x: x["ex_date"] or "", reverse=True)
    logger.info(f"{ticker}: {len(dividends)} distributions scraped from Roundhill")
    return dividends, None


def scrape_multiple_roundhill_etfs(tickers: list[str]) -> dict[str, list[dict]]:
    """
    Scrape multiple ETFs efficiently, sharing a single session + token.

    Returns:
        { ticker: [dividend_dicts, ...], ... }
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    token = _get_token(session)
    if not token:
        logger.error("Could not get Roundhill token — returning empty results")
        return {t.upper(): [] for t in tickers}

    results = {}
    for ticker in tickers:
        ticker = ticker.upper()
        if ticker not in ETF_URL_MAPPING:
            results[ticker] = []
            continue

        session.headers["Referer"] = f"{BASE_URL}/etf/{ETF_URL_MAPPING[ticker]}/"
        try:
            data = {
                "upperetf": ticker,
                "loweretf": ticker.lower() + "/",
                "token":    token,
                "is_ajax":  "1",
            }
            r = session.post(DISTRI_URL, data=data, timeout=15)
            r.raise_for_status()
            raw = json.loads(r.text)
        except Exception as e:
            logger.error(f"{ticker}: scrape failed — {e}")
            results[ticker] = []
            continue

        dividends = []
        rows = raw.values() if isinstance(raw, dict) else raw
        for row in rows:
            try:
                if row[0] == "Declaration":
                    continue
                amount = float(row[4])
                if amount <= 0:
                    continue
                dividends.append({
                    "declaration": _parse_date(row[0]),
                    "ex_date":     _parse_date(row[1]),
                    "record_date": _parse_date(row[2]),
                    "pay_date":    _parse_date(row[3]),
                    "amount":      round(amount, 6),
                })
            except (IndexError, ValueError, TypeError):
                continue

        dividends.sort(key=lambda x: x["ex_date"] or "", reverse=True)
        results[ticker] = dividends
        logger.info(f"{ticker}: {len(dividends)} distributions")

    return results


def get_latest_dividend(ticker: str) -> Optional[dict]:
    """Return only the most recent distribution for a ticker, or None."""
    divs, _ = scrape_roundhill_dividends(ticker)
    return divs[0] if divs else None


def get_all_roundhill_etfs() -> list[str]:
    return list(ETF_URL_MAPPING.keys())


def is_roundhill_etf(ticker: str) -> bool:
    return ticker.upper() in ETF_URL_MAPPING


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test_tickers = ["NVDW", "HOOW", "ARMW", "AVGW", "AMDW", "GOOW", "AMZW"]
    print(f"Scraping {len(test_tickers)} ETFs...\n")
    results = scrape_multiple_roundhill_etfs(test_tickers)

    for ticker, divs in results.items():
        if divs:
            latest = divs[0]
            print(f"{ticker:6s}  {len(divs):3d} entries  "
                  f"latest: ex={latest['ex_date']}  "
                  f"pay={latest['pay_date']}  "
                  f"amt=${latest['amount']:.6f}")
        else:
            print(f"{ticker:6s}  no data")
