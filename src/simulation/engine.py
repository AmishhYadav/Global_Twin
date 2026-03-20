"""
Global Twin — Simulation Engine (v2.0)

Hybrid approach: ML predictions + structural economic multipliers.
The ML models learn from data, but with synthetic data the cross-variable
correlations are weak. Structural multipliers encode known real-world
economic relationships to ensure realistic cascade behavior.
"""

import pandas as pd
import numpy as np
from src.features.build_features import create_time_series_features


# ── Structural Economic Multipliers ──
# Format: (source, target): multiplier
# A multiplier of 0.3 means: if source changes +10%, target changes +3%
# Negative means inverse relationship
STRUCTURAL_MULTIPLIERS = {
    # Oil affects everything
    ("CRUDE_OIL", "GOLD"):              +0.25,  # Oil up → Gold up (inflation hedge)
    ("CRUDE_OIL", "SP500"):             -0.15,  # Oil up → Stocks down (higher costs)
    ("CRUDE_OIL", "US_GDP_GROWTH"):     -0.10,  # Oil up → GDP slows
    ("CRUDE_OIL", "US_CPI_INFLATION"):  +0.30,  # Oil up → Inflation up
    ("CRUDE_OIL", "INR_USD"):           +0.20,  # Oil up → Rupee weakens (India imports oil)
    ("CRUDE_OIL", "EU_CPI_INFLATION"):  +0.25,  # Oil up → EU inflation up
    ("CRUDE_OIL", "IN_CPI_INFLATION"):  +0.35,  # Oil up → India inflation up (heavy importer)
    ("CRUDE_OIL", "NATURAL_GAS"):       +0.40,  # Oil up → Gas up (substitutes)
    ("CRUDE_OIL", "BALTIC_DRY_INDEX"):  +0.30,  # Oil up → Shipping costs up
    
    # US Fed Rate affects markets
    ("US_FED_RATE", "SP500"):           -0.20,  # Rates up → Stocks down
    ("US_FED_RATE", "GOLD"):            -0.15,  # Rates up → Gold down (bonds compete)
    ("US_FED_RATE", "INR_USD"):         +0.25,  # Rates up → USD strong → Rupee weak
    ("US_FED_RATE", "EUR_USD"):         -0.15,  # Rates up → Dollar strong → EUR/USD down
    ("US_FED_RATE", "US_GDP_GROWTH"):   -0.10,  # Rates up → Growth slows

    # Fear index
    ("VIX", "SP500"):                   -0.30,  # Fear up → Stocks crash
    ("VIX", "GOLD"):                    +0.20,  # Fear up → Gold (safe haven)
    
    # GDP affects related vars
    ("US_GDP_GROWTH", "SP500"):         +0.25,  # US growth → Stocks up
    ("US_GDP_GROWTH", "US_CPI_INFLATION"): +0.15, # Growth → Inflation
    ("CN_GDP_GROWTH", "CRUDE_OIL"):     +0.20,  # China growth → Oil demand up
    ("CN_GDP_GROWTH", "COPPER"):        +0.35,  # China growth → Copper up
    ("IN_GDP_GROWTH", "INR_USD"):       -0.15,  # India growth → Rupee strengthens
    
    # Stock market effects
    ("SP500", "VIX"):                   -0.40,  # Stocks down → Fear spikes
    ("SP500", "GOLD"):                  -0.10,  # Stocks down → Gold up (flight to safety)
    
    # Inflation effects
    ("US_CPI_INFLATION", "GOLD"):       +0.20,  # Inflation → Gold hedge
    ("US_CPI_INFLATION", "US_FED_RATE"):+0.15,  # Inflation → Fed raises rates

    # Commodities
    ("GOLD", "INR_USD"):                +0.10,  # Gold up → Rupee weaker (India imports gold)
    ("COPPER", "CN_GDP_GROWTH"):        +0.10,  # Copper indicator of China activity
}


def _ensure_datetime_index(df):
    """Ensure DataFrame has a DatetimeIndex."""
    if 'Date' in df.columns:
        df = df.set_index('Date')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(end='2023-12-31', periods=len(df), freq='D')
    return df


def _apply_structural_cascade(current_values, shocks_applied, all_columns):
    """
    Apply structural economic multipliers to cascade shock effects.
    
    Returns dict of additional changes to apply.
    """
    cascade_changes = {}
    
    for (source, target), multiplier in STRUCTURAL_MULTIPLIERS.items():
        if source in shocks_applied and target in all_columns:
            source_pct = shocks_applied[source]  # e.g., 0.60 for +60%
            cascade_pct = source_pct * multiplier
            
            if target in cascade_changes:
                cascade_changes[target] += cascade_pct  # accumulate from multiple sources
            else:
                cascade_changes[target] = cascade_pct
    
    return cascade_changes


def run_simulation(models_dict, base_df, shock_node=None, shock_pct=None,
                   shocks=None, horizon=3, scenario_name=None):
    """
    Run hybrid simulation: structural multipliers + ML predictions.
    
    The structural multipliers handle cross-variable cascade effects
    (since synthetic data doesn't capture real correlations well).
    ML predictions handle temporal evolution (how each variable evolves over time).
    """
    if shocks is None:
        if shock_node and shock_pct is not None:
            shocks = {shock_node: shock_pct}
        else:
            raise ValueError("Provide either 'shocks' dict or 'shock_node'+'shock_pct'.")
    
    df_base = _ensure_datetime_index(base_df.copy())
    df_shock = _ensure_datetime_index(base_df.copy())
    
    # Apply direct shocks at last timestamp
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
    
    # Apply structural cascade effects at last timestamp
    cascade = _apply_structural_cascade(df_shock.iloc[-1], shocks, df_shock.columns)
    cascade_applied = {}
    for var, cascade_pct in cascade.items():
        if var in df_shock.columns and var not in shocks:  # Don't override direct shocks
            original = df_base.loc[last_idx, var]
            df_shock.loc[last_idx, var] = original * (1 + cascade_pct)
            cascade_applied[var] = {
                "original": float(original),
                "shocked": float(df_shock.loc[last_idx, var]),
                "pct": cascade_pct,
                "cascade": True,
            }
    
    # Merge into applied_shocks for reporting
    applied_shocks.update(cascade_applied)
    
    base_trajectories = []
    shock_trajectories = []
    
    # Track cumulative shocks for cascading through steps
    cumulative_shocks = dict(shocks)
    cumulative_shocks.update(cascade)
    
    for step in range(1, horizon + 1):
        # Regenerate features
        feat_base = create_time_series_features(df_base).iloc[-1:]
        feat_shock = create_time_series_features(df_shock).iloc[-1:]
        
        step_base_preds = {}
        step_shock_preds = {}
        
        for target, model_data in models_dict.items():
            model = model_data['model']
            features = model_data['feature_names']
            
            for f in features:
                if f not in feat_base.columns:
                    feat_base[f] = 0.0
                if f not in feat_shock.columns:
                    feat_shock[f] = 0.0
            
            b_pred = model.predict(feat_base[features])[0]
            s_pred = model.predict(feat_shock[features])[0]
            
            step_base_preds[target] = float(b_pred)
            step_shock_preds[target] = float(s_pred)
        
        # Apply structural cascade to predictions (compound effect over steps)
        cascade_step = _apply_structural_cascade(step_shock_preds, cumulative_shocks, models_dict.keys())
        for var, cascade_pct in cascade_step.items():
            if var in step_shock_preds and var not in shocks:
                base_val = step_base_preds.get(var, step_shock_preds[var])
                # Increasing cascade over steps (compounding)
                step_shock_preds[var] = base_val * (1 + cascade_pct * (step / horizon))
        
        # Append predictions for next step
        next_date = df_base.index[-1] + pd.Timedelta(days=1)
        
        new_base = df_base.iloc[-1].copy()
        new_shock = df_shock.iloc[-1].copy()
        new_base.name = next_date
        new_shock.name = next_date
        
        for k, v in step_base_preds.items():
            new_base[k] = v
        for k, v in step_shock_preds.items():
            new_shock[k] = v
        
        # Persist shocked values for directly shocked variables
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
