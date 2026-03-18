import pandas as pd
from src.features.build_features import create_time_series_features

def run_simulation(models_dict, base_df, shock_node, shock_pct, horizon=3):
    """
    Run temporal cascade simulation natively through sliding ML variables for T+x horizon.
    
    Args:
        models_dict: Output of train_models
        base_df: Cleaned dataframe tracking real historicals (used to build sliding windows)
        shock_node: Column name to shock natively
        shock_pct: Percentage to adjust (e.g. 0.20)
        horizon: T+ steps out to forecast
    """
    # Defensive chronological copies
    df_base = base_df.copy()
    df_shock = base_df.copy()
    
    # Identify the exact last index in df_shock (which is theoretically T=0 today)
    last_idx = df_shock.index[-1]
    
    # Apply initial percentage shock directly to timeline parameter
    df_shock.loc[last_idx, shock_node] = df_shock.loc[last_idx, shock_node] * (1 + shock_pct)
    
    base_trajectories = []
    shock_trajectories = []
    
    for step in range(1, horizon + 1):
        # 1. Regenerate features completely fresh (dropping NAs built into the function)
        feat_base = create_time_series_features(df_base).iloc[-1:]
        feat_shock = create_time_series_features(df_shock).iloc[-1:]
        
        step_base_preds = {}
        step_shock_preds = {}
        
        # 2. Extract ML predictions for step
        for target, model_data in models_dict.items():
            rf = model_data['model']
            features = model_data['feature_names']
            
            # Predict
            b_pred = rf.predict(feat_base[features])[0]
            s_pred = rf.predict(feat_shock[features])[0]
            
            step_base_preds[target] = b_pred
            step_shock_preds[target] = s_pred
            
        # 3. Append generated T+step target variables BACK into the dataframe
        next_date = df_base.index[-1] + pd.Timedelta(days=1)
        
        new_row_base = df_base.iloc[-1].copy()
        new_row_shock = df_shock.iloc[-1].copy()
        new_row_base.name = next_date
        new_row_shock.name = next_date
        
        for k, v in step_base_preds.items():
            new_row_base[k] = v
        for k, v in step_shock_preds.items():
            new_row_shock[k] = v
            
        # Optional: Maintain shock state persistence across T+1, 2, 3 so the array 
        # treats the new structural price as permanent for independent variables that weren't ML evaluated
        new_row_shock[shock_node] = df_shock.loc[df_shock.index[-1], shock_node]
            
        df_base = pd.concat([df_base, pd.DataFrame([new_row_base])])
        df_shock = pd.concat([df_shock, pd.DataFrame([new_row_shock])])
        
        base_trajectories.append(step_base_preds)
        shock_trajectories.append(step_shock_preds)
        
    return {
        'baseline': base_trajectories,
        'shocked': shock_trajectories,
        'df_base': df_base,
        'df_shock': df_shock
    }
