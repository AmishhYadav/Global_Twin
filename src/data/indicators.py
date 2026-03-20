"""
Global Twin — Indicator Registry

Defines all tracked economic indicators across 5 economies.
Each indicator maps to a data source (FRED or Yahoo Finance), 
a series/ticker ID, and metadata for normalization.
"""

# ─────────────────────────────────────────────
# FRED Series IDs (free, no key needed for < 120 req/min)
# ─────────────────────────────────────────────
FRED_INDICATORS = {
    # United States
    "US_GDP_GROWTH":          {"series": "A191RL1Q225SBEA", "freq": "quarterly", "desc": "US Real GDP Growth Rate (%)"},
    "US_UNEMPLOYMENT":        {"series": "UNRATE",          "freq": "monthly",   "desc": "US Unemployment Rate (%)"},
    "US_CPI_INFLATION":       {"series": "CPIAUCSL",        "freq": "monthly",   "desc": "US Consumer Price Index"},
    "US_FED_RATE":            {"series": "FEDFUNDS",        "freq": "monthly",   "desc": "US Federal Funds Rate (%)"},
    "US_TRADE_BALANCE":       {"series": "BOPGSTB",         "freq": "monthly",   "desc": "US Trade Balance (Millions $)"},
    "US_CONSUMER_CONFIDENCE": {"series": "UMCSENT",         "freq": "monthly",   "desc": "US Consumer Sentiment Index"},
    "US_MANUFACTURING_PMI":   {"series": "MANEMP",          "freq": "monthly",   "desc": "US Manufacturing Employment (proxy PMI)"},
    
    # European Union / Eurozone
    "EU_GDP_GROWTH":          {"series": "CLVMNACSCAB1GQEA19", "freq": "quarterly", "desc": "Euro Area Real GDP"},
    "EU_UNEMPLOYMENT":        {"series": "LRHUTTTTEZM156S",    "freq": "monthly",   "desc": "Euro Area Unemployment Rate (%)"},
    "EU_CPI_INFLATION":       {"series": "CP0000EZ19M086NEST", "freq": "monthly",   "desc": "Euro Area HICP"},
    "EU_ECB_RATE":            {"series": "ECBDFR",             "freq": "monthly",   "desc": "ECB Deposit Facility Rate (%)"},
    
    # China
    "CN_GDP_GROWTH":          {"series": "MKTGDPCNA646NWDB", "freq": "annual",    "desc": "China GDP (current USD)"},
    "CN_CPI_INFLATION":       {"series": "FPCPITOTLZGCHN",   "freq": "annual",    "desc": "China CPI Inflation (%)"},
    
    # India
    "IN_GDP_GROWTH":          {"series": "MKTGDPINA646NWDB", "freq": "annual",    "desc": "India GDP (current USD)"},
    "IN_CPI_INFLATION":       {"series": "FPCPITOTLZGIND",   "freq": "annual",    "desc": "India CPI Inflation (%)"},
    
    # Japan
    "JP_GDP_GROWTH":          {"series": "MKTGDPJPA646NWDB", "freq": "annual",    "desc": "Japan GDP (current USD)"},
    "JP_UNEMPLOYMENT":        {"series": "LRHUTTTTJPM156S",   "freq": "monthly",   "desc": "Japan Unemployment Rate (%)"},
    "JP_CPI_INFLATION":       {"series": "FPCPITOTLZGJPN",   "freq": "annual",    "desc": "Japan CPI Inflation (%)"},
    "JP_BOJ_RATE":            {"series": "IRSTCB01JPM156N",   "freq": "monthly",   "desc": "Japan Central Bank Rate (%)"},
}

# ─────────────────────────────────────────────
# Yahoo Finance Tickers (market data, commodities, FX)
# ─────────────────────────────────────────────
YAHOO_INDICATORS = {
    # Commodities
    "CRUDE_OIL":          {"ticker": "CL=F",   "desc": "WTI Crude Oil Futures ($/barrel)"},
    "NATURAL_GAS":        {"ticker": "NG=F",   "desc": "Natural Gas Futures ($/MMBtu)"},
    "GOLD":               {"ticker": "GC=F",   "desc": "Gold Futures ($/oz)"},
    "COPPER":             {"ticker": "HG=F",   "desc": "Copper Futures ($/lb)"},
    "WHEAT":              {"ticker": "ZW=F",   "desc": "Wheat Futures ($/bushel)"},
    
    # Exchange Rates (all vs USD)
    "EUR_USD":            {"ticker": "EURUSD=X", "desc": "EUR/USD Exchange Rate"},
    "CNY_USD":            {"ticker": "CNY=X",    "desc": "USD/CNY Exchange Rate"},
    "INR_USD":            {"ticker": "INR=X",    "desc": "USD/INR Exchange Rate"},
    "JPY_USD":            {"ticker": "JPY=X",    "desc": "USD/JPY Exchange Rate"},
    
    # Indices (economic proxies)
    "SP500":              {"ticker": "^GSPC",    "desc": "S&P 500 Index"},
    "BALTIC_DRY_INDEX":   {"ticker": "^BDI",     "desc": "Baltic Dry Index (shipping cost proxy)"},
    "VIX":                {"ticker": "^VIX",     "desc": "CBOE Volatility Index (fear gauge)"},
    "SEMICONDUCTOR_IDX":  {"ticker": "^SOX",     "desc": "PHLX Semiconductor Index"},
}

# ─────────────────────────────────────────────
# Country → indicator mapping for easy lookup
# ─────────────────────────────────────────────
COUNTRY_INDICATORS = {
    "US": [k for k in FRED_INDICATORS if k.startswith("US_")],
    "EU": [k for k in FRED_INDICATORS if k.startswith("EU_")],
    "CN": [k for k in FRED_INDICATORS if k.startswith("CN_")],
    "IN": [k for k in FRED_INDICATORS if k.startswith("IN_")],
    "JP": [k for k in FRED_INDICATORS if k.startswith("JP_")],
}

# Global indicators available to all countries
GLOBAL_INDICATORS = list(YAHOO_INDICATORS.keys())

COUNTRIES = {
    "US": "United States",
    "EU": "European Union",
    "CN": "China",
    "IN": "India",
    "JP": "Japan",
}

def get_all_indicator_names():
    """Return flat list of all indicator names."""
    return list(FRED_INDICATORS.keys()) + list(YAHOO_INDICATORS.keys())

def get_indicator_info(name):
    """Get metadata for a single indicator by name."""
    if name in FRED_INDICATORS:
        return {**FRED_INDICATORS[name], "source": "FRED"}
    elif name in YAHOO_INDICATORS:
        return {**YAHOO_INDICATORS[name], "source": "YAHOO"}
    return None
