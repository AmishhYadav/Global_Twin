#!/usr/bin/env python3
"""
Global Twin — Data Download Script

Run this to fetch all real-world economic indicators:
    python scripts/fetch_data.py

Optional arguments:
    --start 2015-01-01   (default: 2018-01-01)
    --output data/raw    (default: data/raw)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fetch import fetch_all_indicators
from src.data.indicators import (
    FRED_INDICATORS, YAHOO_INDICATORS, COUNTRIES,
    COUNTRY_INDICATORS, GLOBAL_INDICATORS
)


def main():
    parser = argparse.ArgumentParser(description="Fetch Global Twin economic data")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🌍 GLOBAL TWIN — DATA FETCHER")
    print("=" * 60)
    print(f"\n  Start date:  {args.start}")
    print(f"  Output dir:  {args.output}")
    print(f"  FRED indicators:  {len(FRED_INDICATORS)}")
    print(f"  Yahoo indicators: {len(YAHOO_INDICATORS)}")
    print(f"  Total:            {len(FRED_INDICATORS) + len(YAHOO_INDICATORS)}")
    print(f"\n  Countries: {', '.join(f'{v} ({k})' for k, v in COUNTRIES.items())}")
    
    print("\n  Country-specific indicators:")
    for code, indicators in COUNTRY_INDICATORS.items():
        print(f"    {code}: {', '.join(indicators)}")
    print(f"    GLOBAL: {', '.join(GLOBAL_INDICATORS)}")
    
    print()
    
    result = fetch_all_indicators(start_date=args.start, save_dir=args.output)
    
    fred_df = result['fred']
    yahoo_df = result['yahoo']
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    
    if not fred_df.empty:
        print(f"\n  FRED data: {fred_df.shape[0]} days × {fred_df.shape[1]} indicators")
        print(f"  Date range: {fred_df.index.min().date()} to {fred_df.index.max().date()}")
    
    if not yahoo_df.empty:
        print(f"\n  Yahoo data: {yahoo_df.shape[0]} days × {yahoo_df.shape[1]} indicators")
        print(f"  Date range: {yahoo_df.index.min().date()} to {yahoo_df.index.max().date()}")
    
    print(f"\n  Files saved to: {args.output}/")
    print("  → fred_indicators.csv")
    print("  → yahoo_indicators.csv")
    print("\n  Next: Run the dashboard with 'streamlit run src/dashboard/app.py'")


if __name__ == "__main__":
    main()
