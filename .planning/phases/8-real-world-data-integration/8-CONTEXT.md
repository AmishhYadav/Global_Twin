# Phase 8: Real-World Data Integration - Context

**Gathered:** 2026-03-20
**Status:** Executed

## Implementation Decisions
- Use FRED's public CSV endpoint (no API key required) for macro indicators.
- Use yfinance for commodities, FX, and market indices.
- Forward-fill mixed frequencies (monthly/quarterly/annual) to daily.
- Save raw data as CSVs in `data/raw/` for reproducibility.

## Indicators (30 total)
- **FRED (20):** GDP Growth, Unemployment, CPI, Central Bank Rates, Trade Balance, Consumer Confidence, Manufacturing — per country.
- **Yahoo (10):** Crude Oil, Natural Gas, Gold, Copper, Wheat, EUR/USD, CNY/USD, INR/USD, JPY/USD, S&P 500, Baltic Dry Index, VIX, Semiconductor Index.

## Countries
US, EU, China, India, Japan
