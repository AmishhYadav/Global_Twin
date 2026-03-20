#!/usr/bin/env python3
"""
Test: Multi-Model ML Pipeline (Phase 11)

Validates RF + GBR training, auto-selection, and model registry.
Run: python scripts/test_ml_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.data.country_manager import CountryDataManager
from src.features.build_features import build_full_feature_matrix
from src.models.train import train_models, get_comparison_report
from src.models.registry import save_registry, load_registry, list_registry


def main():
    print("=" * 60)
    print("  PHASE 11: Multi-Model ML Pipeline Test")
    print("=" * 60)
    
    # 1. Load synthetic data and build features
    print("\n[Step 1] Loading data + building features...")
    mgr = CountryDataManager()
    mgr.load_synthetic()
    df = mgr.get_all_data()
    feat_df = build_full_feature_matrix(df, lags=[1, 3], rolling_windows=[7])
    
    # 2. Train multi-model pipeline
    print("\n[Step 2] Training multi-model pipeline...")
    targets = ['CRUDE_OIL', 'US_CPI_INFLATION', 'SP500']
    results = train_models(feat_df, targets, verbose=True)
    
    # 3. Validate results structure
    print("\n[Step 3] Validating results structure...")
    for target in targets:
        r = results[target]
        assert 'model' in r, f"Missing model for {target}"
        assert 'model_name' in r, f"Missing model_name for {target}"
        assert r['model_name'] in ['RandomForest', 'GradientBoosting']
        assert 'all_candidates' in r, f"Missing candidates for {target}"
        assert len(r['all_candidates']) == 2, f"Expected 2 candidates for {target}"
        assert 'feature_importances' in r
        print(f"  ✓ {target}: {r['model_name']} selected, {len(r['feature_names'])} features")
    
    # 4. Comparison report
    print("\n[Step 4] Comparison Report:")
    report = get_comparison_report(results)
    print(report)
    
    # 5. Save to registry
    print("\n[Step 5] Saving to model registry...")
    registry_dir = "/tmp/test_registry"
    save_registry(results, registry_dir=registry_dir)
    
    # 6. Load from registry
    print("\n[Step 6] Loading from registry...")
    loaded = load_registry(registry_dir=registry_dir)
    for target in targets:
        assert target in loaded
        assert loaded[target]['model'] is not None
        # Verify loaded model can predict
        sample = feat_df[loaded[target]['feature_names']].iloc[-1:].values
        pred = loaded[target]['model'].predict(sample)
        print(f"  ✓ {target}: loaded & predicted successfully (pred={pred[0]:.4f})")
    
    # 7. List registry
    print("\n[Step 7] Registry listing:")
    list_registry(registry_dir=registry_dir)
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
