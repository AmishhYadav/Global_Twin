#!/usr/bin/env python3
"""
Test: Multi-Country Data Architecture (Phase 9)

Validates the CountryDataManager with synthetic data fallback.
Run: python scripts/test_countries.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.country_manager import CountryDataManager
from src.data.indicators import COUNTRIES


def main():
    print("=" * 60)
    print("  PHASE 9: Multi-Country Data Architecture Test")
    print("=" * 60)
    
    # Initialize with synthetic data (no API calls needed)
    mgr = CountryDataManager()
    mgr.load_synthetic()
    
    # 1. Summary
    print("\n[Test 1] Data Summary")
    mgr.summary()
    
    # 2. Per-country access
    print("\n[Test 2] Per-Country DataFrames")
    for code, name in COUNTRIES.items():
        df = mgr.get_country_data(code)
        assert not df.empty, f"{code} DataFrame is empty!"
        print(f"  ✓ {name} ({code}): {df.shape[1]} columns, {df.shape[0]} rows")
        print(f"    Indicators: {', '.join(df.columns[:5])}...")
    
    # 3. Single indicator query
    print("\n[Test 3] Single Indicator Query")
    oil = mgr.get_indicator('CRUDE_OIL')
    assert len(oil) > 1000, "Oil series too short"
    print(f"  ✓ CRUDE_OIL: {len(oil)} observations, mean=${oil.mean():.2f}")
    
    us_unemp = mgr.get_indicator('US_UNEMPLOYMENT')
    print(f"  ✓ US_UNEMPLOYMENT: {len(us_unemp)} observations, mean={us_unemp.mean():.2f}%")
    
    # 4. Cross-country comparison
    print("\n[Test 4] Cross-Country Comparison")
    gdp_compare = mgr.get_cross_country('GDP_GROWTH')
    print(f"  ✓ GDP_GROWTH across {list(gdp_compare.columns)}:")
    for country in gdp_compare.columns:
        print(f"    {country}: mean={gdp_compare[country].mean():.2f}")
    
    inflation_compare = mgr.get_cross_country('CPI_INFLATION')
    print(f"  ✓ CPI_INFLATION across {list(inflation_compare.columns)}")
    
    # 5. List indicators
    print("\n[Test 5] Indicator Listing")
    us_indicators = mgr.list_indicators('US')
    print(f"  ✓ US indicators ({len(us_indicators)}): {us_indicators[:5]}...")
    
    all_indicators = mgr.list_indicators()
    print(f"  ✓ All indicators: {len(all_indicators)} total")
    
    # 6. Full unified access
    print("\n[Test 6] Unified DataFrame")
    full = mgr.get_all_data()
    assert full.shape[1] >= 25, f"Expected 25+ indicators, got {full.shape[1]}"
    print(f"  ✓ Full data: {full.shape[0]} rows × {full.shape[1]} columns")
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
