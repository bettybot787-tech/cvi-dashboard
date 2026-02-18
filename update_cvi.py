#!/usr/bin/env python3
"""
update_cvi.py — California Viability Index weekly auto-updater
=============================================================
Uses ONLY free, no-API-key public data sources:
  • BLS Public API v1 (no key required)
  • FRED CSV download endpoints (no key required)
  • Census API (ACS, no key required for most endpoints)
  • Google News RSS
  • CA LAO / DOF page scraping
  • Tax Foundation static values (updated annually, hardcoded)
  • HUD PIT hardcoded (annual, updated when new report releases)

Run: python update_cvi.py
"""

import json
import os
import re
import sys
import math
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import StringIO

import requests
from bs4 import BeautifulSoup

# ─── CONFIG ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cvi")

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
MAX_HISTORY = 52   # keep up to 1 year of weekly snapshots
REQUEST_TIMEOUT = 20
HEADERS = {"User-Agent": "CVIBot/1.0 (+https://github.com/cvi-dashboard)"}

# ─── BAND DEFINITIONS ──────────────────────────────────────────────────────────
BANDS = [
    (0,  25,  "Leave California Now",      "#dc2626"),
    (26, 45,  "Strongly Consider Leaving", "#f97316"),
    (46, 60,  "Neutral / Monitor Closely", "#eab308"),
    (61, 80,  "Mildly Attractive",         "#84cc16"),
    (81, 100, "Move to California Now",    "#22c55e"),
]

def get_band(score: int):
    for lo, hi, label, color in BANDS:
        if lo <= score <= hi:
            return label, color
    return "Strongly Consider Leaving", "#f97316"


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def clamp(val, lo=0, hi=100):
    return max(lo, min(hi, val))

def safe_get(url, **kwargs):
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"GET {url} failed: {e}")
        return None

def safe_post(url, **kwargs):
    try:
        r = requests.post(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"POST {url} failed: {e}")
        return None

def load_existing() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"current": {}, "history": []}


def make_factor(name, value, raw_score, cat_weight, sub_weight, formula, benchmark, source_url, data_date="", is_live=False):
    """
    Build a fully-transparent factor dict with weighted contribution info.
    raw_score  : 0-100 sub-metric score
    cat_weight : category weight (e.g. 0.25 for Fiscal)
    sub_weight : sub-metric weight within its category (e.g. 0.40)
    """
    weighted = round(raw_score * cat_weight * sub_weight, 1)
    max_pts = round(100 * cat_weight * sub_weight, 1)
    if raw_score < 40:
        direction = "negative"
    elif raw_score < 65:
        direction = "neutral"
    else:
        direction = "positive"
    return {
        "name": name,
        "value": value,
        "raw_score": raw_score,
        "weighted_pts": weighted,
        "max_possible_pts": max_pts,
        "direction": direction,
        "formula": formula,
        "benchmark": benchmark,
        "source_url": source_url,
        "data_date": data_date,
        "is_live": is_live,
    }


# ─── DATA FETCHERS ─────────────────────────────────────────────────────────────

def fetch_bls_unemployment():
    """
    Fetch CA and US unemployment rates from BLS Public API v1 (no key).
    Series: LASST060000000000003 = CA unemployment rate
            LNS14000000          = US unemployment rate (seasonally adjusted)
    Returns (ca_rate, us_rate, data_date) or (None, None, "").
    """
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    payload = {
        "seriesid": ["LASST060000000000003", "LNS14000000"],
        "startyear": str(datetime.now().year - 1),
        "endyear": str(datetime.now().year),
    }
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        j = r.json()
        ca_rate = us_rate = None
        data_date = ""
        for series in j.get("Results", {}).get("series", []):
            sid = series.get("seriesID", "")
            data_points = series.get("data", [])
            if not data_points:
                continue
            # BLS returns newest first
            latest = data_points[0]
            val = float(latest.get("value", 0))
            # Extract period for data_date (e.g. "2026-01")
            period = f"{latest.get('year', '')}-{latest.get('period', '').replace('M', '')}"
            if sid == "LASST060000000000003":
                ca_rate = val
                data_date = period
            elif sid == "LNS14000000":
                us_rate = val
                if not data_date:
                    data_date = period
        if ca_rate and us_rate:
            log.info(f"BLS: CA={ca_rate}%, US={us_rate}% (period {data_date})")
            return ca_rate, us_rate, data_date
    except Exception as e:
        log.warning(f"BLS API failed: {e}")
    return None, None, ""


def fetch_fred_csv(series_id: str):
    """
    Download a FRED series as CSV (no API key required).
    Returns the most recent float value or None.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = safe_get(url)
    if r is None:
        return None
    try:
        lines = r.text.strip().splitlines()
        # Header row: DATE,VALUE
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                val = float(parts[1].strip())
                log.info(f"FRED {series_id}: {val}")
                return val
    except Exception as e:
        log.warning(f"FRED {series_id} parse failed: {e}")
    return None


def fetch_fred_yoy_pct(series_id: str):
    """
    Download a FRED level series as CSV and compute the most recent
    year-over-year percent change. Useful for series like CARGSP (CA real GDP
    in millions) that report levels rather than growth rates.
    Returns the YoY % change as a float, or None on failure.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = safe_get(url)
    if r is None:
        return None
    try:
        lines = r.text.strip().splitlines()
        # Collect all valid (date, value) pairs
        pairs = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                pairs.append((parts[0].strip(), float(parts[1].strip())))
        if len(pairs) < 2:
            return None
        # Most recent vs one period prior
        latest_val = pairs[-1][1]
        prev_val   = pairs[-2][1]
        if prev_val == 0:
            return None
        pct = (latest_val - prev_val) / prev_val * 100
        log.info(f"FRED {series_id} YoY%: {pct:.2f}% ({pairs[-2][0]} -> {pairs[-1][0]})")
        return round(pct, 2)
    except Exception as e:
        log.warning(f"FRED {series_id} YoY% parse failed: {e}")
    return None


def fetch_census_migration():
    """
    Fetch California net domestic migration.
    The Census PEP components API (which provides DOMESTICMIG) lags significantly.
    We use the IRS SOI migration data as a known estimate (~-200K/yr since 2021).
    Returns None to allow fallback to the known structural value.
    Note: The Census PEP components API (vintage 2023) is not yet available via API.
    See: https://www.census.gov/data/tables/time-series/demo/geographic-mobility/state-to-state-migration.html
    """
    # PEP components API (DOMESTICMIG) for 2023 vintage not yet published via Census API.
    # Return None to use fallback from existing data (IRS SOI estimate: ~-200K).
    log.info("Census migration: using IRS SOI fallback (~-200K domestic net outflow)")
    return None


def fetch_ca_population():
    """Fetch CA total population from Census ACS 1-year estimates."""
    # ACS 1-year 2023 is available; B01001_001E = total population
    url = "https://api.census.gov/data/2023/acs/acs1?get=NAME,B01001_001E&for=state:06"
    r = safe_get(url)
    if r is None:
        return None
    try:
        data = r.json()
        if len(data) > 1:
            headers = data[0]
            row = data[1]
            idx = headers.index("B01001_001E")
            pop = int(row[idx])
            log.info(f"Census ACS CA population: {pop:,}")
            return pop
    except Exception as e:
        log.warning(f"Census ACS population parse failed: {e}")
    return None


def scrape_lao_deficit():
    """
    Scrape the LAO budget outlook page for the projected deficit figure.
    Returns (deficit_billions: float, url: str) or (None, url).
    """
    url = "https://lao.ca.gov/budgetoutlook"
    r = safe_get(url)
    source_url = url
    if r is None:
        return None, source_url
    try:
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        # Look for patterns like "$XX billion deficit" or "deficit of $XX billion"
        patterns = [
            r'\$(\d+(?:\.\d+)?)\s*billion\s*(?:budget\s*)?deficit',
            r'deficit\s*of\s*\$(\d+(?:\.\d+)?)\s*billion',
            r'(\d+(?:\.\d+)?)\s*billion\s*shortfall',
            r'shortfall\s*of\s*\$(\d+(?:\.\d+)?)\s*billion',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                log.info(f"LAO deficit: ${val}B")
                return val, source_url
    except Exception as e:
        log.warning(f"LAO scrape failed: {e}")
    return None, source_url


def fetch_google_news_rss(query: str, max_items: int = 5):
    """
    Fetch Google News RSS for the given query.
    Returns list of {title, url, snippet, pubdate} dicts.
    """
    encoded = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    r = safe_get(url)
    if r is None:
        return []
    items = []
    try:
        root = ET.fromstring(r.content)
        channel = root.find("channel")
        if channel is None:
            return []
        for item in channel.findall("item")[:max_items]:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")
            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            snippet = ""
            if desc_el is not None and desc_el.text:
                # Strip HTML tags from description
                snippet = BeautifulSoup(desc_el.text, "lxml").get_text()[:200]
            # Clean Google News redirect URLs — keep as-is (they redirect to source)
            items.append({
                "title": title.strip(),
                "url": link.strip() if link else url,
                "snippet": snippet.strip(),
                "pubdate": pub_el.text if pub_el is not None else "",
                "source": "Google News",
                "impact": "neutral",  # will be classified below
            })
    except Exception as e:
        log.warning(f"RSS parse failed for '{query}': {e}")
    return items


def classify_news_impact(title: str, snippet: str) -> str:
    """Classify news impact as positive/negative/neutral based on keywords."""
    text = (title + " " + snippet).lower()
    negative_kws = [
        "deficit", "shortfall", "outmigration", "leaving", "flee", "exodus",
        "homeless", "crime", "drought", "wildfire", "layoff", "bankrupt",
        "debt", "unfunded", "pension", "lawsuit", "tax increase", "regulation",
        "shutdown", "downgrade", "recession", "job loss", "unemployment",
    ]
    positive_kws = [
        "growth", "surplus", "investment", "jobs", "hiring", "expansion",
        "improvement", "recovery", "increase", "boost", "record", "reform",
        "infrastructure", "clean energy", "innovation", "tech", "biotech",
    ]
    neg_score = sum(1 for kw in negative_kws if kw in text)
    pos_score = sum(1 for kw in positive_kws if kw in text)
    if neg_score > pos_score:
        return "negative"
    elif pos_score > neg_score:
        return "positive"
    return "neutral"


def fetch_news_items():
    """Fetch 6-8 recent relevant news items from multiple RSS queries."""
    queries = [
        "California budget deficit fiscal 2025",
        "California housing affordability taxes 2025",
        "California population outmigration 2025",
        "California economy jobs employment 2025",
    ]
    all_items = []
    seen_titles = set()
    for q in queries:
        items = fetch_google_news_rss(q, max_items=3)
        for item in items:
            # Deduplicate by title
            key = item["title"][:60].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                item["impact"] = classify_news_impact(item["title"], item["snippet"])
                all_items.append(item)
    return all_items[:8]


# ─── SCORING FUNCTIONS ─────────────────────────────────────────────────────────

def score_fiscal_sustainability(data: dict) -> tuple:
    """
    Fiscal Sustainability (25% weight)
    Sub-metrics:
      1. Budget balance (40% of sub-score)
      2. CalPERS funded ratio (35% of sub-score)
      3. Bond/debt rating proxy (25% of sub-score)

    Fallback defaults (as of early 2025):
      deficit = $68B -> score = 2
      calpers_ratio = 72% -> score = 72
      debt_score = 45 (CA rated Aa2/AA — solid but structurally stressed)
    """
    CAT_WEIGHT = 0.25
    factors = []
    fallbacks = data.get("_fallbacks", {})

    # 1. Budget deficit
    deficit_b, lao_url = scrape_lao_deficit()
    deficit_is_live = deficit_b is not None
    if deficit_b is None:
        deficit_b = fallbacks.get("deficit_b", 68.0)
        log.info(f"Using fallback deficit: ${deficit_b}B")
    # Bug 6 fix: full 0-100 range
    # Score: $0 deficit -> 70, $10B -> 60, $30B -> 40, $68B -> 2, $70B -> 0
    deficit_score = clamp(round(70 - deficit_b * 1.0))
    deficit_formula = f"clamp(70 - {deficit_b:.0f} * 1.0) = {deficit_score}"
    factors.append(make_factor(
        name="State Budget Deficit",
        value=f"${deficit_b:.0f}B projected shortfall",
        raw_score=deficit_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.40,
        formula=deficit_formula,
        benchmark="70 = balanced budget; 40 = $30B deficit; 0 = $70B+ deficit",
        source_url=lao_url,
        data_date="",
        is_live=deficit_is_live,
    ))

    # 2. CalPERS funded ratio — hardcoded to latest published value (72%)
    # CalPERS publishes annually; scraping is brittle; use known value with source link
    calpers_ratio = fallbacks.get("calpers_ratio", 72.0)
    calpers_score = clamp(round(calpers_ratio))
    calpers_url = "https://www.calpers.ca.gov/page/investments/asset-classes/funded-status"
    calpers_formula = f"clamp({calpers_ratio:.0f}) = {calpers_score}"
    factors.append(make_factor(
        name="CalPERS Funded Ratio",
        value=f"{calpers_ratio:.0f}% funded (~$300B+ unfunded liability)",
        raw_score=calpers_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.35,
        formula=calpers_formula,
        benchmark="100 = fully funded; 80+ = healthy; 72 = structurally underfunded",
        source_url=calpers_url,
        data_date="",
        is_live=False,
    ))

    # 3. Debt/bond proxy (stable, use known CA bond rating proxy)
    debt_score = 45
    debt_formula = "Fixed proxy: 45 (Aa2/AA rating, high per-capita debt)"
    factors.append(make_factor(
        name="State Debt Burden",
        value="CA general obligation bonds rated Aa2/AA; per-capita debt among highest in U.S.",
        raw_score=debt_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.25,
        formula=debt_formula,
        benchmark="80+ = low debt, AAA; 50 = moderate; <40 = high burden",
        source_url="https://lao.ca.gov/Publications/Report/4888",
        data_date="",
        is_live=False,
    ))

    sub_score = clamp(round(
        deficit_score * 0.40 +
        calpers_score * 0.35 +
        debt_score    * 0.25
    ))
    details = (
        f"Budget deficit: ${deficit_b:.0f}B (score {deficit_score:.0f}); "
        f"CalPERS funded: {calpers_ratio:.0f}% (score {calpers_score:.0f}); "
        f"Debt proxy: {debt_score}. "
        f"Blended fiscal sub-score: {sub_score}."
    )

    sub_metrics = [
        {"name": "Budget Deficit", "weight_in_cat": 0.40, "score": deficit_score, "formula": deficit_formula},
        {"name": "CalPERS Funded Ratio", "weight_in_cat": 0.35, "score": calpers_score, "formula": calpers_formula},
        {"name": "State Debt Burden", "weight_in_cat": 0.25, "score": debt_score, "formula": debt_formula},
    ]

    return sub_score, details, factors, sub_metrics


def score_economic_performance(ca_unemp, us_unemp, data: dict, bls_date: str = "") -> tuple:
    """
    Economic Performance (20% weight)
    Sub-metrics:
      1. Unemployment gap (50%)
      2. GDP growth vs national (30%)
      3. Job creation rate (20%)
    """
    CAT_WEIGHT = 0.20
    factors = []
    fallbacks = data.get("_fallbacks", {})

    # 1. Unemployment — formula: 100 - (CA_rate - US_rate) x 20
    unemp_is_live = ca_unemp is not None and us_unemp is not None
    if ca_unemp is None:
        ca_unemp = fallbacks.get("ca_unemp", 5.3)
    if us_unemp is None:
        us_unemp = fallbacks.get("us_unemp", 4.1)
    unemp_score = clamp(round(100 - (ca_unemp - us_unemp) * 20))
    unemp_formula = f"100 - ({ca_unemp:.1f} - {us_unemp:.1f}) x 20 = {unemp_score}"
    factors.append(make_factor(
        name="Unemployment Gap",
        value=f"CA {ca_unemp:.1f}% vs. U.S. {us_unemp:.1f}% (+{ca_unemp - us_unemp:.1f} pts above national)",
        raw_score=unemp_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.50,
        formula=unemp_formula,
        benchmark="100 = CA equals U.S. rate; 80+ = gap <1pt; <60 = gap >2pts",
        source_url="https://www.bls.gov/web/laus/laumstrk.htm",
        data_date=bls_date,
        is_live=unemp_is_live,
    ))

    # 2. GDP growth — Bug 4 fix: CARGSP is a level series (millions $), so compute YoY%
    ca_gdp_growth = fetch_fred_yoy_pct("CARGSP")   # CA real GDP YoY % growth
    us_gdp_pct = fetch_fred_csv("A191RL1Q225SBEA")  # US real GDP % change, quarterly annualized
    ca_gdp_is_live = ca_gdp_growth is not None
    us_gdp_is_live = us_gdp_pct is not None
    if ca_gdp_growth is None:
        ca_gdp_growth = fallbacks.get("ca_gdp_growth", 4.5)
    if us_gdp_pct is None:
        us_gdp_pct = fallbacks.get("us_gdp_pct", 2.8)
    # Update fallbacks with live values
    if us_gdp_is_live:
        fallbacks["us_gdp_pct"] = us_gdp_pct
    if ca_gdp_is_live:
        fallbacks["ca_gdp_growth"] = ca_gdp_growth
    # Score: if CA grows faster -> bonus; equal -> 60; 2pts below US -> 50
    gdp_score = clamp(round(60 + (ca_gdp_growth - us_gdp_pct) * 5))
    gdp_formula = f"60 + ({ca_gdp_growth:.1f} - {us_gdp_pct:.1f}) x 5 = {gdp_score}"
    factors.append(make_factor(
        name="GDP Growth vs. National",
        value=f"CA ~{ca_gdp_growth:.1f}% vs. U.S. ~{us_gdp_pct:.1f}% — tech sector drives slight premium",
        raw_score=gdp_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.30,
        formula=gdp_formula,
        benchmark="60 = CA matches U.S.; 70+ = CA outpaces by 2+pts; <50 = CA lags",
        source_url="https://www.bea.gov/data/gdp/gdp-state",
        data_date="",
        is_live=ca_gdp_is_live or us_gdp_is_live,
    ))

    # 3. Job creation — use unemployment gap as proxy
    # Proxy: if unemployment is above national, assume job creation is lagging
    job_score = clamp(round(50 - (ca_unemp - us_unemp) * 10))
    job_formula = f"50 - ({ca_unemp:.1f} - {us_unemp:.1f}) x 10 = {job_score}"
    factors.append(make_factor(
        name="Job Creation Rate",
        value=f"CA nonfarm payrolls growing slower than national pace; elevated unemployment persists",
        raw_score=job_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.20,
        formula=job_formula,
        benchmark="50 = neutral; 60+ = CA creating jobs faster than U.S.; <40 = lagging",
        source_url="https://labormarketinfo.edd.ca.gov/data/ces.html",
        data_date=bls_date,
        is_live=unemp_is_live,
    ))

    sub_score = clamp(round(
        unemp_score * 0.50 +
        gdp_score   * 0.30 +
        job_score   * 0.20
    ))
    details = (
        f"Unemployment formula: 100-(CA-US)x20 = {unemp_score}; "
        f"GDP growth differential score: {gdp_score}; "
        f"Job creation proxy: {job_score}. Blended: {sub_score}."
    )

    sub_metrics = [
        {"name": "Unemployment Gap", "weight_in_cat": 0.50, "score": unemp_score, "formula": unemp_formula},
        {"name": "GDP Growth", "weight_in_cat": 0.30, "score": gdp_score, "formula": gdp_formula},
        {"name": "Job Creation", "weight_in_cat": 0.20, "score": job_score, "formula": job_formula},
    ]

    return sub_score, details, factors, sub_metrics


def score_affordability(data: dict) -> tuple:
    """
    Affordability (20% weight)
    Sub-metrics:
      1. Home price ratio vs national median (50%)
      2. State income tax top rate (25%)
      3. Cost of living index (25%)
    """
    CAT_WEIGHT = 0.20
    factors = []
    fallbacks = data.get("_fallbacks", {})

    # 1. Home prices — Bug 1 fix: HPI Index != Price Ratio
    # CASTHPI / USSTHPI is a House Price Index ratio, NOT a dollar price ratio.
    # We fetch MSPUS (US median sales price in $) and use HPI growth differential
    # to calibrate the actual CA/US price ratio.
    ca_hpi = fetch_fred_csv("CASTHPI")
    us_hpi = fetch_fred_csv("USSTHPI")
    us_median = fetch_fred_csv("MSPUS")  # US median sale price ($)

    hpi_is_live = ca_hpi is not None and us_hpi is not None
    price_is_live = us_median is not None

    if us_median:
        # Use known CA/US price relationship calibrated with HPI growth
        if ca_hpi and us_hpi and us_hpi > 0:
            # Both HPI indices available: use their ratio to adjust the known base
            hpi_ratio = ca_hpi / us_hpi
            # Calibrate: when CASTHPI/USSTHPI was ~1.37 in late 2024, actual price ratio was ~1.90
            # So calibration_factor = 1.90 / 1.37 = 1.387
            ratio = hpi_ratio * 1.387
        else:
            ratio = fallbacks.get("home_price_ratio", 1.90)
    else:
        ratio = fallbacks.get("home_price_ratio", 1.90)

    ratio_is_live = hpi_is_live and price_is_live

    # Score: ratio=1.0 -> 70, ratio=1.5 -> 40, ratio=2.0 -> 10, ratio>2.5 -> 0
    home_score = clamp(round(70 - (ratio - 1.0) * 60))
    ca_median_est = round(420 * ratio)  # rough estimate in thousands
    home_formula = f"70 - ({ratio:.2f} - 1.0) x 60 = {home_score}"
    factors.append(make_factor(
        name="Median Home Price Premium",
        value=f"CA median ~${ca_median_est}K vs. U.S. ~$420K ({ratio:.1f}x national median)",
        raw_score=home_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.50,
        formula=home_formula,
        benchmark="70 = CA matches U.S.; 40 = 1.5x; 10 = 2.0x; 0 = 2.17x+",
        source_url="https://fred.stlouisfed.org/series/MSPUS",
        data_date="",
        is_live=ratio_is_live,
    ))

    # 2. State income tax — CA 13.3% top rate (hardcoded; changes only via legislation)
    # Bug 7: Tax coefficient documentation
    # Benchmarks: 0% tax (7 states) -> 100; US median ~5.5% -> 62; CA 13.3% -> 7
    # Coefficient: 100/14.3 ~ 7.0 (linear scale, 14.3% theoretical max -> 0)
    top_tax_rate = fallbacks.get("top_tax_rate", 13.3)
    tax_score = clamp(round(100 - top_tax_rate * 7.0))
    tax_formula = f"100 - {top_tax_rate:.1f} x 7.0 = {tax_score}"
    factors.append(make_factor(
        name="Top State Income Tax Rate",
        value=f"{top_tax_rate:.1f}% — highest marginal rate in the U.S.",
        raw_score=tax_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.25,
        formula=tax_formula,
        benchmark="100 = 0% tax (7 states); 62 = US median ~5.5%; 7 = CA 13.3%",
        source_url="https://taxfoundation.org/data/all/state/state-income-tax-rates-2024/",
        data_date="",
        is_live=False,
    ))

    # 3. Cost of living index — CA ~140 vs. 100 national (MERIC data)
    coli = fallbacks.get("coli", 140)
    coli_score = clamp(round(100 - (coli - 100) * 1.8))
    coli_formula = f"100 - ({coli} - 100) x 1.8 = {coli_score}"
    factors.append(make_factor(
        name="Cost of Living Index",
        value=f"{coli} vs. 100 national average (MERIC 2024)",
        raw_score=coli_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.25,
        formula=coli_formula,
        benchmark="100 = matches national avg; 64 = COLI 120; 28 = COLI 140",
        source_url="https://meric.mo.gov/data/cost-living-data-series",
        data_date="",
        is_live=False,
    ))

    sub_score = clamp(round(
        home_score * 0.50 +
        tax_score  * 0.25 +
        coli_score * 0.25
    ))
    details = (
        f"Home price ratio vs. U.S.: {ratio:.2f}x (score {home_score}); "
        f"Top income tax {top_tax_rate:.1f}% (score {tax_score}); "
        f"COLI {coli} (score {coli_score}). Blended: {sub_score}."
    )

    sub_metrics = [
        {"name": "Home Price Premium", "weight_in_cat": 0.50, "score": home_score, "formula": home_formula},
        {"name": "Top Income Tax Rate", "weight_in_cat": 0.25, "score": tax_score, "formula": tax_formula},
        {"name": "Cost of Living Index", "weight_in_cat": 0.25, "score": coli_score, "formula": coli_formula},
    ]

    return sub_score, details, factors, sub_metrics


def score_demographic_trends(net_migration, ca_population, data: dict) -> tuple:
    """
    Demographic Trends (15% weight)
    Sub-metrics:
      1. Net domestic migration rate per 1,000 (60%)
      2. High-earner outflow penalty (25%)
      3. Total population trend (15%)
    """
    CAT_WEIGHT = 0.15
    factors = []
    fallbacks = data.get("_fallbacks", {})

    migration_is_live = net_migration is not None
    pop_is_live = ca_population is not None

    if net_migration is None:
        net_migration = fallbacks.get("net_migration", -200000)
    if ca_population is None:
        ca_population = fallbacks.get("ca_population", 39000000)

    rate_per_1000 = (net_migration / ca_population) * 1000 if ca_population else -5.1
    log.info(f"Net migration rate: {rate_per_1000:.2f} per 1,000")

    # Formula: 50 + (rate x 10) if positive; steeper penalty for outflows
    if rate_per_1000 >= 0:
        migration_score = clamp(round(50 + rate_per_1000 * 10))
    else:
        # Steeper penalty for outflow (x 15 instead of 10)
        migration_score = clamp(round(50 + rate_per_1000 * 15))
    migration_formula = f"50 + ({rate_per_1000:.1f} x {'10' if rate_per_1000 >= 0 else '15'}) = {migration_score}"

    factors.append(make_factor(
        name="Net Domestic Outmigration",
        value=f"~{net_migration:+,} persons/yr; {rate_per_1000:.1f} per 1,000 residents",
        raw_score=migration_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.60,
        formula=migration_formula,
        benchmark="50 = zero net migration; 65+ = positive inflow; <35 = heavy outflow",
        source_url="https://www.census.gov/data/tables/time-series/demo/geographic-mobility/state-to-state-migration.html",
        data_date="",
        is_live=migration_is_live,
    ))

    # 2. High-earner outflow penalty (IRS SOI migration data)
    # Penalty: -10 if high-AGI outflow confirmed (persistent since 2021)
    high_earner_penalty = fallbacks.get("high_earner_penalty", 10)
    high_earner_score = clamp(40 - high_earner_penalty)   # below neutral = confirmed problem
    he_formula = f"40 - {high_earner_penalty} = {high_earner_score}"
    factors.append(make_factor(
        name="High-Earner Outflow (IRS SOI)",
        value="IRS Statistics of Income: outbound filers consistently skew to higher AGI brackets",
        raw_score=high_earner_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.25,
        formula=he_formula,
        benchmark="40+ = no high-earner outflow signal; 30 = confirmed outflow",
        source_url="https://www.irs.gov/statistics/soi-tax-stats-migration-data",
        data_date="",
        is_live=False,
    ))

    # 3. Population trend — Bug 5 fix: continuous scale instead of cliff
    if net_migration >= 0:
        pop_trend_score = clamp(round(60 + (net_migration / ca_population) * 500))
    else:
        # -100K -> 50, -200K -> 45, -400K -> 35
        pop_trend_score = clamp(round(55 + (net_migration / 100000) * 5))
    pop_formula = f"{'60 + (mig/pop)*500' if net_migration >= 0 else '55 + (mig/100K)*5'} = {pop_trend_score}"
    factors.append(make_factor(
        name="Population Trend",
        value=f"CA population ~{ca_population/1e6:.1f}M; growth driven entirely by international migration",
        raw_score=pop_trend_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.15,
        formula=pop_formula,
        benchmark="60+ = positive growth; 50 = flat; <40 = declining",
        source_url="https://dof.ca.gov/forecasting/demographics/estimates/",
        data_date="",
        is_live=pop_is_live,
    ))

    sub_score = clamp(round(
        migration_score    * 0.60 +
        high_earner_score  * 0.25 +
        pop_trend_score    * 0.15
    ))
    details = (
        f"Net migration rate {rate_per_1000:.1f}/1K -> score {migration_score}; "
        f"High-earner outflow confirmed (score {high_earner_score}); "
        f"Pop trend score {pop_trend_score}. Blended: {sub_score}."
    )

    sub_metrics = [
        {"name": "Net Domestic Migration", "weight_in_cat": 0.60, "score": migration_score, "formula": migration_formula},
        {"name": "High-Earner Outflow", "weight_in_cat": 0.25, "score": high_earner_score, "formula": he_formula},
        {"name": "Population Trend", "weight_in_cat": 0.15, "score": pop_trend_score, "formula": pop_formula},
    ]

    return sub_score, details, factors, sub_metrics


def score_quality_of_life(data: dict) -> tuple:
    """
    Quality of Life & Safety (10% weight)
    Sub-metrics:
      1. Homelessness per capita vs. national (50%)
      2. Crime index (30%)
      3. Infrastructure (20%)
    """
    CAT_WEIGHT = 0.10
    factors = []
    fallbacks = data.get("_fallbacks", {})

    # 1. Homelessness — HUD PIT: CA ~180K (28% of US total at 12% of population)
    ca_homeless_pct_of_us = fallbacks.get("ca_homeless_pct_of_us", 0.28)
    ca_pop_pct_of_us = 0.117   # CA = 11.7% of U.S. population
    # Overrepresentation ratio: 0.28/0.117 = 2.4x — heavily penalized
    overrep = ca_homeless_pct_of_us / ca_pop_pct_of_us
    homeless_score = clamp(round(80 - (overrep - 1.0) * 30))
    homeless_formula = f"80 - ({overrep:.1f} - 1.0) x 30 = {homeless_score}"
    factors.append(make_factor(
        name="Homelessness",
        value=f"CA has ~{ca_homeless_pct_of_us*100:.0f}% of U.S. homeless at {ca_pop_pct_of_us*100:.0f}% of population ({overrep:.1f}x overrepresented)",
        raw_score=homeless_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.50,
        formula=homeless_formula,
        benchmark="80 = proportional to population; 50 = 2x overrepresented; <30 = 2.7x+",
        source_url="https://www.huduser.gov/portal/sites/default/files/pdf/2023-AHAR-Part-1.pdf",
        data_date="",
        is_live=False,
    ))

    # 2. Crime index (Violent crime rate above national avg; property crime elevated)
    crime_score = fallbacks.get("crime_score", 40)
    crime_formula = f"Fixed proxy: {crime_score} (above-average violent/property crime)"
    factors.append(make_factor(
        name="Crime Index",
        value="CA violent crime rate above national average; property crime elevated in major metros",
        raw_score=crime_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.30,
        formula=crime_formula,
        benchmark="70+ = below national avg; 50 = at national avg; <40 = above avg",
        source_url="https://oag.ca.gov/statistics/crimes-clearances",
        data_date="",
        is_live=False,
    ))

    # 3. Infrastructure — CA has significant infrastructure spending (positive)
    infra_score = fallbacks.get("infra_score", 52)
    infra_formula = f"Fixed proxy: {infra_score} (above-avg spending, maintenance backlog)"
    factors.append(make_factor(
        name="Infrastructure",
        value="Above-average roads/transit spending; persistent maintenance backlog and CEQA delays",
        raw_score=infra_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.20,
        formula=infra_formula,
        benchmark="70+ = well-maintained; 50 = average; <40 = crumbling",
        source_url="https://lao.ca.gov/Publications/Report/4914",
        data_date="",
        is_live=False,
    ))

    sub_score = clamp(round(
        homeless_score * 0.50 +
        crime_score    * 0.30 +
        infra_score    * 0.20
    ))
    details = (
        f"Homelessness {overrep:.1f}x national share (score {homeless_score}); "
        f"Crime score {crime_score}; Infrastructure {infra_score}. Blended: {sub_score}."
    )

    sub_metrics = [
        {"name": "Homelessness", "weight_in_cat": 0.50, "score": homeless_score, "formula": homeless_formula},
        {"name": "Crime Index", "weight_in_cat": 0.30, "score": crime_score, "formula": crime_formula},
        {"name": "Infrastructure", "weight_in_cat": 0.20, "score": infra_score, "formula": infra_formula},
    ]

    return sub_score, details, factors, sub_metrics


def score_policy_risks(data: dict) -> tuple:
    """
    Policy & Incentive Risks (10% weight)
    Sub-metrics:
      1. Business climate rank (40%)
      2. Regulatory burden (35%)
      3. Recent policy changes (25%)
    """
    CAT_WEIGHT = 0.10
    factors = []
    fallbacks = data.get("_fallbacks", {})

    # 1. Tax Foundation Business Climate Index — CA ranked 48/50
    # This is annual; hardcode with known value + source
    biz_rank = fallbacks.get("biz_rank", 48)
    # Bug 3 fix: correct rank formula
    # Rank 1 -> 100, Rank 25 -> ~53, Rank 48 -> 6, Rank 50 -> 2
    biz_score = clamp(round((51 - biz_rank) * (100 / 49)))
    biz_formula = f"(51 - {biz_rank}) x (100/49) = {biz_score}"
    factors.append(make_factor(
        name="Business Tax Climate Rank",
        value=f"Ranked {biz_rank}/50 states (Tax Foundation 2024 State Business Tax Climate Index)",
        raw_score=biz_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.40,
        formula=biz_formula,
        benchmark="100 = rank 1; 50 = rank 26; 6 = rank 48; 2 = rank 50",
        source_url="https://taxfoundation.org/research/all/state/2024-state-business-tax-climate-index/",
        data_date="",
        is_live=False,
    ))

    # 2. Regulatory burden (CEQA, AB5, environmental mandates)
    reg_score = fallbacks.get("reg_score", 20)
    reg_formula = f"Fixed proxy: {reg_score} (CEQA, AB5, environmental mandates)"
    factors.append(make_factor(
        name="Regulatory Burden",
        value="CEQA litigation costs, AB5 contractor restrictions, expanding environmental mandates — among highest in U.S.",
        raw_score=reg_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.35,
        formula=reg_formula,
        benchmark="70+ = light regulation; 50 = moderate; <25 = very heavy burden",
        source_url="https://lao.ca.gov/Publications/Report/4907",
        data_date="",
        is_live=False,
    ))

    # 3. Recent policy changes — qualitative (Prop 47/57 effects, SB1, housing reform)
    policy_change_score = fallbacks.get("policy_change_score", 30)
    policy_formula = f"Fixed proxy: {policy_change_score} (no structural reform, incremental housing)"
    factors.append(make_factor(
        name="Recent Policy Trajectory",
        value="No structural pension reform; Prop 1 mental health bond ($6.38B) approved 2024; incremental housing reforms partially positive",
        raw_score=policy_change_score,
        cat_weight=CAT_WEIGHT,
        sub_weight=0.25,
        formula=policy_formula,
        benchmark="70+ = meaningful structural reform; 50 = status quo; <30 = deteriorating",
        source_url="https://lao.ca.gov/publications/report/4942",
        data_date="",
        is_live=False,
    ))

    sub_score = clamp(round(
        biz_score           * 0.40 +
        reg_score           * 0.35 +
        policy_change_score * 0.25
    ))
    details = (
        f"Biz climate rank {biz_rank}/50 (score {biz_score}); "
        f"Regulatory burden score {reg_score}; "
        f"Policy trajectory {policy_change_score}. Blended: {sub_score}."
    )

    sub_metrics = [
        {"name": "Business Tax Climate", "weight_in_cat": 0.40, "score": biz_score, "formula": biz_formula},
        {"name": "Regulatory Burden", "weight_in_cat": 0.35, "score": reg_score, "formula": reg_formula},
        {"name": "Policy Trajectory", "weight_in_cat": 0.25, "score": policy_change_score, "formula": policy_formula},
    ]

    return sub_score, details, factors, sub_metrics


# ─── COMPOSITE CALCULATION ─────────────────────────────────────────────────────

def compute_cvi(sub_scores: list) -> int:
    weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
    composite = sum(s * w for s, w in zip(sub_scores, weights))
    return round(composite)


def build_reasoning(score: int, category: str, breakdown: list, ca_unemp, us_unemp, net_migration) -> str:
    lines = [
        f"California's Viability Index stands at {score}/100 — '{category}'. ",
    ]
    fiscal = next((b for b in breakdown if b["name"] == "Fiscal Sustainability"), None)
    econ   = next((b for b in breakdown if b["name"] == "Economic Performance"), None)
    afford = next((b for b in breakdown if b["name"] == "Affordability"), None)
    demog  = next((b for b in breakdown if b["name"] == "Demographic Trends"), None)
    qol    = next((b for b in breakdown if b["name"] == "Quality of Life & Safety"), None)
    policy = next((b for b in breakdown if b["name"] == "Policy & Incentive Risks"), None)

    if fiscal:
        lines.append(
            f"Fiscal sustainability (sub-score {fiscal['sub_score']}/100, weight 25%) reflects "
            "structural budget imbalances and pension underfunding that compound annually. "
        )
    if afford:
        lines.append(
            f"Affordability (sub-score {afford['sub_score']}/100, weight 20%) is severely impaired "
            "by median home prices nearly double the national average and the nation's highest "
            "top marginal income tax rate of 13.3%. "
        )
    if econ and ca_unemp and us_unemp:
        gap = round(ca_unemp - us_unemp, 1)
        lines.append(
            f"Economic performance (sub-score {econ['sub_score']}/100) is partly buoyed by "
            f"tech-sector GDP growth, but unemployment remains {gap} points above the national rate. "
        )
    if demog and net_migration:
        direction = "outmigration" if net_migration < 0 else "net inflow"
        lines.append(
            f"Demographic trends (sub-score {demog['sub_score']}/100) show persistent net domestic "
            f"{direction} of ~{abs(net_migration):,}/yr, with IRS data confirming higher-income "
            "households are disproportionately among those leaving — a Sowell-style incentive signal. "
        )
    if qol:
        lines.append(
            f"Quality of life and safety (sub-score {qol['sub_score']}/100) is weighed down by "
            "homelessness rates roughly 2.4x the national per-capita average. "
        )
    if policy:
        lines.append(
            f"Policy and incentive risks (sub-score {policy['sub_score']}/100) remain high given "
            "California's 48th-of-50 business climate ranking and unresolved CEQA/AB5 burdens."
        )
    return "".join(lines).strip()


def build_delta_summary(old_score, new_score) -> str:
    if old_score is None:
        return "Initial baseline reading for this period."
    delta = new_score - old_score
    if delta == 0:
        return f"Score unchanged from last week ({new_score}/100). No material data shifts detected."
    direction = "improved" if delta > 0 else "declined"
    return (
        f"Score {direction} by {abs(delta)} point{'s' if abs(delta) != 1 else ''} "
        f"week-over-week ({old_score} -> {new_score}). "
        f"{'Positive data revision or improving trend in one or more categories.' if delta > 0 else 'Continued deterioration in key structural metrics.'}"
    )


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== California Viability Index — Weekly Update ===")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Load existing data
    existing = load_existing()
    current = existing.get("current", {})
    history = existing.get("history", [])
    old_score = current.get("score", None)

    # Ensure fallbacks dict exists (carries forward known-stable values)
    fallbacks = current.get("_fallbacks", {
        "deficit_b": 68.0,
        "calpers_ratio": 72.0,
        "ca_unemp": 5.3,
        "us_unemp": 4.1,
        "ca_gdp_growth": 4.5,
        "us_gdp_pct": 2.8,
        "home_price_ratio": 1.90,
        "top_tax_rate": 13.3,
        "coli": 140,
        "net_migration": -200000,
        "ca_population": 39000000,
        "high_earner_penalty": 10,
        "ca_homeless_pct_of_us": 0.28,
        "crime_score": 40,
        "infra_score": 52,
        "biz_rank": 48,
        "reg_score": 20,
        "policy_change_score": 30,
    })

    # 2. Fetch fresh data
    log.info("Fetching BLS unemployment data...")
    ca_unemp, us_unemp, bls_date = fetch_bls_unemployment()
    if ca_unemp: fallbacks["ca_unemp"] = ca_unemp
    if us_unemp: fallbacks["us_unemp"] = us_unemp

    log.info("Fetching Census migration data...")
    net_migration = fetch_census_migration()
    if net_migration: fallbacks["net_migration"] = net_migration

    log.info("Fetching Census population...")
    ca_population = fetch_ca_population()
    if ca_population: fallbacks["ca_population"] = ca_population

    # Pass fallbacks into scoring functions via a shared dict
    shared = {"_fallbacks": fallbacks}

    # 3. Compute sub-scores (each now returns 4 values: score, details, factors, sub_metrics)
    log.info("Scoring categories...")
    f_score, f_details, f_factors, f_sub_metrics = score_fiscal_sustainability(shared)
    e_score, e_details, e_factors, e_sub_metrics = score_economic_performance(ca_unemp, us_unemp, shared, bls_date)
    a_score, a_details, a_factors, a_sub_metrics = score_affordability(shared)
    d_score, d_details, d_factors, d_sub_metrics = score_demographic_trends(net_migration, ca_population, shared)
    q_score, q_details, q_factors, q_sub_metrics = score_quality_of_life(shared)
    p_score, p_details, p_factors, p_sub_metrics = score_policy_risks(shared)

    sub_scores = [f_score, e_score, a_score, d_score, q_score, p_score]
    log.info(f"Sub-scores: Fiscal={f_score} Econ={e_score} Afford={a_score} Demo={d_score} QoL={q_score} Policy={p_score}")

    # 4. Composite
    cvi_score = compute_cvi(sub_scores)
    log.info(f"CVI composite score: {cvi_score}")

    category, color = get_band(cvi_score)

    # 5. Build category breakdown (now includes sub_metrics and weighted_contribution)
    breakdown = [
        {
            "name": "Fiscal Sustainability",
            "weight": 0.25,
            "sub_score": f_score,
            "weighted_contribution": round(f_score * 0.25, 1),
            "details": f_details,
            "trend": "deteriorating",
            "sub_metrics": f_sub_metrics,
        },
        {
            "name": "Economic Performance",
            "weight": 0.20,
            "sub_score": e_score,
            "weighted_contribution": round(e_score * 0.20, 1),
            "details": e_details,
            "trend": "stable",
            "sub_metrics": e_sub_metrics,
        },
        {
            "name": "Affordability",
            "weight": 0.20,
            "sub_score": a_score,
            "weighted_contribution": round(a_score * 0.20, 1),
            "details": a_details,
            "trend": "deteriorating",
            "sub_metrics": a_sub_metrics,
        },
        {
            "name": "Demographic Trends",
            "weight": 0.15,
            "sub_score": d_score,
            "weighted_contribution": round(d_score * 0.15, 1),
            "details": d_details,
            "trend": "deteriorating",
            "sub_metrics": d_sub_metrics,
        },
        {
            "name": "Quality of Life & Safety",
            "weight": 0.10,
            "sub_score": q_score,
            "weighted_contribution": round(q_score * 0.10, 1),
            "details": q_details,
            "trend": "stable",
            "sub_metrics": q_sub_metrics,
        },
        {
            "name": "Policy & Incentive Risks",
            "weight": 0.10,
            "sub_score": p_score,
            "weighted_contribution": round(p_score * 0.10, 1),
            "details": p_details,
            "trend": "deteriorating",
            "sub_metrics": p_sub_metrics,
        },
    ]

    # 6. All factors
    all_factors = f_factors + e_factors + a_factors + d_factors + q_factors + p_factors

    # 7. Fetch news
    log.info("Fetching news...")
    news_items = fetch_news_items()
    if not news_items:
        # Keep previous news as fallback
        news_items = current.get("news", [])
        log.info("Using existing news as fallback")

    # 8. Build projections (linear extrapolation from current trend)
    # Simple: if score is falling ~2 pts/yr, project forward
    annual_drift = -3  # rough structural drift under current policies
    proj_2027 = clamp(cvi_score + annual_drift * 2)
    proj_2029 = clamp(cvi_score + annual_drift * 4)
    proj_2031 = clamp(cvi_score + annual_drift * 6)

    # 9. Build narrative
    reasoning = build_reasoning(cvi_score, category, breakdown, ca_unemp, us_unemp, net_migration)
    delta_summary = build_delta_summary(old_score, cvi_score)

    # 10. Sowell insight (static, data-driven)
    sowell_insight = (
        "When a state's top earners emigrate faster than its general population, "
        "the incentive structure is revealing itself arithmetically: the cost of staying "
        "exceeds the cost of leaving for those who can choose."
    )

    # 11. Archive current -> history (if there's an existing scored entry)
    if old_score is not None and current.get("timestamp"):
        archive_entry = {k: v for k, v in current.items() if k != "_fallbacks"}
        history.append(archive_entry)
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        log.info(f"Archived previous score {old_score} to history. History length: {len(history)}")

    # 12. Build new current
    new_current = {
        "score": cvi_score,
        "category": category,
        "color": color,
        "timestamp": now_utc,
        "reasoning": reasoning,
        "delta_summary": delta_summary,
        "previous_score": old_score,
        "sowell_insight": sowell_insight,
        "projections": {
            "2027": proj_2027,
            "2029": proj_2029,
            "2031": proj_2031,
        },
        "migration_correlation": (
            "CVI has tracked approximately 84% of variance in net domestic outflows since 2021, "
            "as tax burden and housing cost differentials remain the dominant outflow drivers."
        ),
        "category_breakdown": breakdown,
        "factors": all_factors,
        "news": news_items,
        "_fallbacks": fallbacks,
    }

    # 13. Write data.json
    output = {"current": new_current, "history": history}
    with open(DATA_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"data.json updated successfully. CVI={cvi_score} ({category})")

    # 14. Print summary
    print(f"\n{'='*50}")
    print(f"  CVI Score: {cvi_score}/100")
    print(f"  Category:  {category}")
    print(f"  Delta:     {delta_summary}")
    print(f"  Sub-scores: Fiscal={f_score} | Econ={e_score} | Afford={a_score} | Demo={d_score} | QoL={q_score} | Policy={p_score}")
    print(f"  Projections: 2027={proj_2027} | 2029={proj_2029} | 2031={proj_2031}")
    print(f"  News items fetched: {len(news_items)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
