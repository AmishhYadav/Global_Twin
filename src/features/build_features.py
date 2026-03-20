"""
Global Twin — Enhanced Feature Engineering (v2.0)

Builds a rich feature matrix supporting:
  1. Standard time-series features (lags, rolling stats, momentum)
  2. Cross-country lag correlations (e.g., US rate → INR/USD)
  3. Global composite indicators (weighted GDP, risk index)
  4. Inter-sector dependency signals (commodity → inflation spreads)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  1. Core Time-Series Features (v1.0 enhanced)
# ─────────────────────────────────────────────

def create_time_series_features(df, lags=[1, 3, 7], rolling_windows=[7, 14]):
    """
    Generate standard time-series features for all numeric columns.
    
    Features per column:
        - Lag values (shift by N days)
        - Rolling mean
        - Rolling std (volatility)
        - Percentage rate of change
        - Momentum (current - lag)
    
    Args:
        df: DataFrame with DatetimeIndex, numeric columns only.
        lags: List of lag periods in days.
        rolling_windows: List of rolling window sizes.
    
    Returns:
        pd.DataFrame with original + engineered features, NaN rows dropped.
    """
    df_feat = df.copy()
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        for lag in lags:
            df_feat[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)
            # Momentum: difference between current and lagged value
            df_feat[f"{col}_momentum_{lag}"] = df_feat[col] - df_feat[col].shift(lag)
        
        for w in rolling_windows:
            df_feat[f"{col}_rmean_{w}"] = df_feat[col].rolling(window=w).mean()
            df_feat[f"{col}_rstd_{w}"] = df_feat[col].rolling(window=w).std()
            df_feat[f"{col}_roc_{w}"] = df_feat[col].pct_change(periods=w)
    
    df_feat.dropna(inplace=True)
    return df_feat


# ─────────────────────────────────────────────
#  2. Cross-Country Correlation Features
# ─────────────────────────────────────────────

CROSS_COUNTRY_PAIRS = [
    # (driver_indicator, target_indicator, lag_days, feature_name)
    ("US_FED_RATE",      "INR_USD",    [1, 3, 7],  "USrate_to_INR"),
    ("US_FED_RATE",      "EUR_USD",    [1, 3, 7],  "USrate_to_EUR"),
    ("US_FED_RATE",      "JPY_USD",    [1, 3, 7],  "USrate_to_JPY"),
    ("US_FED_RATE",      "CNY_USD",    [1, 3, 7],  "USrate_to_CNY"),
    ("CRUDE_OIL",        "US_CPI_INFLATION",  [7, 14], "Oil_to_USinflation"),
    ("CRUDE_OIL",        "EU_CPI_INFLATION",  [7, 14], "Oil_to_EUinflation"),
    ("CRUDE_OIL",        "IN_CPI_INFLATION",  [7, 14], "Oil_to_INinflation"),
    ("US_UNEMPLOYMENT",  "US_CONSUMER_CONFIDENCE", [1, 7], "USemp_to_USconf"),
    ("CRUDE_OIL",        "BALTIC_DRY_INDEX",  [1, 3],  "Oil_to_Shipping"),
    ("VIX",              "SP500",      [1, 3],  "Fear_to_Market"),
    ("SEMICONDUCTOR_IDX","SP500",      [1, 7],  "Chips_to_Market"),
]


def create_cross_country_features(df):
    """
    Generate lagged cross-country correlation features.
    
    For each defined pair (driver → target), creates lagged driver values
    as predictive features for the target.
    
    Args:
        df: Unified DataFrame with all indicators.
    
    Returns:
        pd.DataFrame with cross-country lag features appended.
    """
    df_feat = df.copy()
    
    for driver, target, lags, name in CROSS_COUNTRY_PAIRS:
        if driver not in df_feat.columns or target not in df_feat.columns:
            continue
        
        for lag in lags:
            df_feat[f"xc_{name}_lag{lag}"] = df_feat[driver].shift(lag)
            # Also add the spread (difference) as a feature
            df_feat[f"xc_{name}_spread_lag{lag}"] = (
                df_feat[driver].shift(lag) - df_feat[target]
            )
    
    return df_feat


# ─────────────────────────────────────────────
#  3. Global Composite Indicators
# ─────────────────────────────────────────────

# GDP weights (approximate share of world GDP)
GDP_WEIGHTS = {
    "US_GDP_GROWTH": 0.25,
    "EU_GDP_GROWTH": 0.18,
    "CN_GDP_GROWTH": 0.18,
    "IN_GDP_GROWTH": 0.07,
    "JP_GDP_GROWTH": 0.05,
}

INFLATION_COLS = [
    "US_CPI_INFLATION", "EU_CPI_INFLATION",
    "CN_CPI_INFLATION", "IN_CPI_INFLATION", "JP_CPI_INFLATION"
]

RISK_COLS = ["VIX", "CRUDE_OIL", "GOLD"]


def create_global_composites(df):
    """
    Generate global composite indicators:
    
    - GLOBAL_GDP_INDEX: Weighted average of GDP growth rates across economies
    - GLOBAL_INFLATION_INDEX: Average inflation across tracked economies
    - GLOBAL_RISK_INDEX: Composite of VIX, Oil volatility, Gold (safe-haven demand)
    - COMMODITY_PRESSURE: Average of energy + metal + agricultural commodity changes
    - USD_STRENGTH: Inverse average of major FX pairs vs USD
    
    Args:
        df: Unified DataFrame with all indicators.
    
    Returns:
        pd.DataFrame with composite columns appended.
    """
    df_comp = df.copy()
    
    # 1. Weighted Global GDP Index
    available_gdp = {k: v for k, v in GDP_WEIGHTS.items() if k in df_comp.columns}
    if available_gdp:
        total_weight = sum(available_gdp.values())
        df_comp['GLOBAL_GDP_INDEX'] = sum(
            df_comp[col] * (w / total_weight) for col, w in available_gdp.items()
        )
    
    # 2. Global Inflation Index (simple average of available CPI indicators)
    avail_inf = [c for c in INFLATION_COLS if c in df_comp.columns]
    if avail_inf:
        df_comp['GLOBAL_INFLATION_INDEX'] = df_comp[avail_inf].mean(axis=1)
    
    # 3. Global Risk Index (normalized composite)
    avail_risk = [c for c in RISK_COLS if c in df_comp.columns]
    if avail_risk:
        # Z-score normalize each risk indicator, then average
        risk_z_scores = pd.DataFrame()
        for col in avail_risk:
            mean = df_comp[col].rolling(window=30, min_periods=1).mean()
            std = df_comp[col].rolling(window=30, min_periods=1).std().replace(0, 1)
            risk_z_scores[col] = (df_comp[col] - mean) / std
        df_comp['GLOBAL_RISK_INDEX'] = risk_z_scores.mean(axis=1)
    
    # 4. Commodity Pressure Index
    commodity_cols = ['CRUDE_OIL', 'NATURAL_GAS', 'GOLD', 'COPPER', 'WHEAT']
    avail_comm = [c for c in commodity_cols if c in df_comp.columns]
    if avail_comm:
        # Average 7-day % change across commodities
        pct_changes = pd.DataFrame()
        for col in avail_comm:
            pct_changes[col] = df_comp[col].pct_change(periods=7)
        df_comp['COMMODITY_PRESSURE'] = pct_changes.mean(axis=1)
    
    # 5. USD Strength Index
    fx_cols = ['EUR_USD', 'CNY_USD', 'INR_USD', 'JPY_USD']
    avail_fx = [c for c in fx_cols if c in df_comp.columns]
    if avail_fx:
        # Normalize each FX pair to 100 at start, then average
        fx_normed = pd.DataFrame()
        for col in avail_fx:
            first_val = df_comp[col].iloc[0]
            if first_val != 0:
                fx_normed[col] = (df_comp[col] / first_val) * 100
        if not fx_normed.empty:
            df_comp['USD_STRENGTH'] = fx_normed.mean(axis=1)
    
    return df_comp


# ─────────────────────────────────────────────
#  4. Inter-Sector Dependency Features
# ─────────────────────────────────────────────

SECTOR_SPREADS = [
    # (col_a, col_b, name) — spread = a - b, ratio = a / b
    ("CRUDE_OIL", "NATURAL_GAS", "energy_spread"),
    ("GOLD", "COPPER", "precious_industrial_ratio"),
    ("SP500", "VIX", "market_fear_spread"),
    ("BALTIC_DRY_INDEX", "CRUDE_OIL", "shipping_energy_ratio"),
]


def create_inter_sector_features(df):
    """
    Generate inter-sector dependency features:
    spreads and ratios between related indicators.
    
    Args:
        df: Unified DataFrame.
    
    Returns:
        pd.DataFrame with sector features appended.
    """
    df_sec = df.copy()
    
    for col_a, col_b, name in SECTOR_SPREADS:
        if col_a not in df_sec.columns or col_b not in df_sec.columns:
            continue
        
        df_sec[f"sector_{name}_spread"] = df_sec[col_a] - df_sec[col_b]
        
        # Safe ratio (avoid division by zero)
        denom = df_sec[col_b].replace(0, np.nan)
        df_sec[f"sector_{name}_ratio"] = df_sec[col_a] / denom
    
    return df_sec


# ─────────────────────────────────────────────
#  5. Master Pipeline
# ─────────────────────────────────────────────

def build_full_feature_matrix(df, lags=[1, 3, 7], rolling_windows=[7, 14]):
    """
    Master feature engineering pipeline. Applies all feature layers:
    
    1. Cross-country lag correlations
    2. Global composite indicators
    3. Inter-sector dependency features
    4. Standard time-series features (lags, rolling stats, momentum)
    
    Args:
        df: Unified DataFrame with DatetimeIndex and all indicators.
        lags: Lag periods for time-series features.
        rolling_windows: Window sizes for rolling features.
    
    Returns:
        pd.DataFrame with comprehensive feature matrix, NaN rows dropped.
    """
    print(f"  [Features] Input: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Layer 1: Cross-country features
    df = create_cross_country_features(df)
    print(f"  [Features] + Cross-country: {df.shape[1]} columns")
    
    # Layer 2: Global composites
    df = create_global_composites(df)
    print(f"  [Features] + Composites: {df.shape[1]} columns")
    
    # Layer 3: Inter-sector features
    df = create_inter_sector_features(df)
    print(f"  [Features] + Inter-sector: {df.shape[1]} columns")
    
    # Layer 4: Standard time-series on all accumulated columns
    df = create_time_series_features(df, lags=lags, rolling_windows=rolling_windows)
    print(f"  [Features] + Time-series: {df.shape[1]} columns (after NaN drop: {df.shape[0]} rows)")
    
    return df
