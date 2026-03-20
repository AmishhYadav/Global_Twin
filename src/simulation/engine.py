"""
Global Twin — Simulation Engine (v2.0)

Supports multi-variable simultaneous shocks from scenarios.
Cascades through the ML pipeline with temporal stepping.
"""

import pandas as pd
import numpy as np
from src.features.build_features import create_time_series_features


def _ensure_datetime_index(df):
    """Ensure DataFrame has a DatetimeIndex."""
    if 'Date' in df.columns:
        df = df.set_index('Date')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(end='2023-12-31', periods=len(df), freq='D')
    return df


def run_simulation(models_dict, base_df, shock_node=None, shock_pct=None,
                   shocks=None, horizon=3, scenario_name=None):
    """
    Run temporal cascade simulation with single or multi-variable shocks.
    
    Args:
        models_dict: Output of train_models.
        base_df: Historical DataFrame (will be copied).
        shock_node: (v1 compat) Single variable to shock.
        shock_pct: (v1 compat) Single shock percentage.
        shocks: dict mapping variable → shock_pct (for multi-variable scenarios).
                Overrides shock_node/shock_pct if provided.
        horizon: T+ steps to forecast.
        scenario_name: Optional label for the scenario.
    
    Returns:
        dict with baseline/shocked trajectories, DataFrames, and metadata.
    """
    # Build shocks dict
    if shocks is None:
        if shock_node and shock_pct is not None:
            shocks = {shock_node: shock_pct}
        else:
            raise ValueError("Provide either 'shocks' dict or 'shock_node'+'shock_pct'.")
    
    df_base = _ensure_datetime_index(base_df.copy())
    df_shock = _ensure_datetime_index(base_df.copy())
    
    # Apply all shocks simultaneously at last timestamp
    last_idx = df_shock.index[-1]
    applied_shocks = {}
    for var, pct in shocks.items():
        if var in df_shock.columns:
            original = df_shock.loc[last_idx, var]
            df_shock.loc[last_idx, var] = original * (1 + pct)
            applied_shocks[var] = {
                "original": float(original),
                "shocked": float(df_shock.loc[last_idx, var]),
                "pct": pct,
            }
    
    base_trajectories = []
    shock_trajectories = []
    
    for step in range(1, horizon + 1):
        # Regenerate features
        feat_base = create_time_series_features(df_base).iloc[-1:]
        feat_shock = create_time_series_features(df_shock).iloc[-1:]
        
        step_base_preds = {}
        step_shock_preds = {}
        
        for target, model_data in models_dict.items():
            model = model_data['model']
            features = model_data['feature_names']
            
            # Fill missing features with 0
            for f in features:
                if f not in feat_base.columns:
                    feat_base[f] = 0.0
                if f not in feat_shock.columns:
                    feat_shock[f] = 0.0
            
            b_pred = model.predict(feat_base[features])[0]
            s_pred = model.predict(feat_shock[features])[0]
            
            step_base_preds[target] = float(b_pred)
            step_shock_preds[target] = float(s_pred)
        
        # Append predictions back for next step
        next_date = df_base.index[-1] + pd.Timedelta(days=1)
        
        new_base = df_base.iloc[-1].copy()
        new_shock = df_shock.iloc[-1].copy()
        new_base.name = next_date
        new_shock.name = next_date
        
        for k, v in step_base_preds.items():
            new_base[k] = v
        for k, v in step_shock_preds.items():
            new_shock[k] = v
        
        # Persist shocked values for origin variables
        for var in shocks:
            if var in df_shock.columns:
                new_shock[var] = df_shock.loc[df_shock.index[-1], var]
        
        df_base = pd.concat([df_base, pd.DataFrame([new_base])])
        df_shock = pd.concat([df_shock, pd.DataFrame([new_shock])])
        
        base_trajectories.append(step_base_preds)
        shock_trajectories.append(step_shock_preds)
    
    return {
        'baseline': base_trajectories,
        'shocked': shock_trajectories,
        'df_base': df_base,
        'df_shock': df_shock,
        'applied_shocks': applied_shocks,
        'horizon': horizon,
        'scenario_name': scenario_name,
    }
