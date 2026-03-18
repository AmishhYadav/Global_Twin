import pandas as pd
import numpy as np

def create_time_series_features(df, lags=[1, 3, 7], rolling_windows=[7, 14]):
    """
    Generate time series features including lags, moving averages, and percentage changes.
    
    Args:
        df: Pandas DataFrame (must contain only numeric columns; Date should be the index or excluded).
        lags: List of days to lag.
        rolling_windows: List of window sizes for moving averages & ROC.
        
    Returns:
        pd.DataFrame with generated features (dropped NaNs).
    """
    df_feat = df.copy()
    
    # Only operate on numeric columns
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        # Lags
        for lag in lags:
            df_feat[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)
            
        # Rolling features
        for w in rolling_windows:
            df_feat[f"{col}_rolling_mean_{w}"] = df_feat[col].rolling(window=w).mean()
            # Percentage Rate of Change (ROC)
            df_feat[f"{col}_pct_change_{w}"] = df_feat[col].pct_change(periods=w)
            
    # Drop rows that have NaNs due to shifting/rolling
    df_feat.dropna(inplace=True)
    return df_feat
