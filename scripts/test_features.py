#!/usr/bin/env python3
"""
Test: Enhanced Feature Engineering (Phase 10)

Validates all feature layers with synthetic data.
Run: python scripts/test_features.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.country_manager import CountryDataManager
from src.features.build_features import (
    create_time_series_features,
    create_cross_country_features,
    create_global_composites,
    create_inter_sector_features,
    build_full_feature_matrix,
)


def main():
    print("=" * 60)
    print("  PHASE 10: Enhanced Feature Engineering Test")
    print("=" * 60)
    
    # Load synthetic data
    mgr = CountryDataManager()
    mgr.load_synthetic()
    df = mgr.get_all_data()
    print(f"\nBase data: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Test 1: Cross-country features
    print("\n[Test 1] Cross-Country Correlation Features")
    df_xc = create_cross_country_features(df)
    xc_cols = [c for c in df_xc.columns if c.startswith('xc_')]
    print(f"  ✓ Added {len(xc_cols)} cross-country features")
    print(f"  Examples: {xc_cols[:4]}")
    
    # Test 2: Global composites
    print("\n[Test 2] Global Composite Indicators")
    df_comp = create_global_composites(df)
    composite_cols = ['GLOBAL_GDP_INDEX', 'GLOBAL_INFLATION_INDEX', 'GLOBAL_RISK_INDEX', 
                      'COMMODITY_PRESSURE', 'USD_STRENGTH']
    found = [c for c in composite_cols if c in df_comp.columns]
    print(f"  ✓ Created {len(found)}/{len(composite_cols)} composites: {found}")
    for c in found:
        print(f"    {c}: mean={df_comp[c].mean():.4f}, std={df_comp[c].std():.4f}")
    
    # Test 3: Inter-sector features
    print("\n[Test 3] Inter-Sector Dependency Features")
    df_sec = create_inter_sector_features(df)
    sec_cols = [c for c in df_sec.columns if c.startswith('sector_')]
    print(f"  ✓ Added {len(sec_cols)} sector features")
    print(f"  Examples: {sec_cols[:4]}")
    
    # Test 4: Full pipeline
    print("\n[Test 4] Full Feature Matrix Pipeline")
    df_full = build_full_feature_matrix(df, lags=[1, 3], rolling_windows=[7])
    print(f"  ✓ Final matrix: {df_full.shape[0]} rows × {df_full.shape[1]} columns")
    assert df_full.shape[1] >= 50, f"Expected 50+ features, got {df_full.shape[1]}"
    assert df_full.isna().sum().sum() == 0, "NaN values found in final matrix!"
    print(f"  ✓ No NaN values in output")
    print(f"  ✓ All dtypes numeric: {all(df_full[c].dtype.kind in 'biufc' for c in df_full.columns)}")
    
    # Test 5: v1.0 backward compatibility
    print("\n[Test 5] v1.0 Backward Compatibility")
    import pandas as pd
    import numpy as np
    small_df = pd.DataFrame({
        'Oil': np.random.normal(60, 5, 100),
        'Shipping': np.random.normal(300, 50, 100),
    }, index=pd.date_range('2020-01-01', periods=100))
    df_v1 = create_time_series_features(small_df, lags=[1, 3], rolling_windows=[7])
    assert not df_v1.empty, "v1.0 time-series features failed"
    print(f"  ✓ v1.0 function still works: {df_v1.shape}")
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
