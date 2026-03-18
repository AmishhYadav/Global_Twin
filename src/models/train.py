import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_models(df, target_cols):
    """
    Train a separate Random Forest Regressor for each target column sequentially using T+1 prediction.
    
    Args:
        df: DataFrame with engineered features.
        target_cols: List of column names to forecast.
    
    Returns:
        dict: Trained models, evaluation metrics, and feature importances mapped by target.
    """
    results = {}
    
    for target in target_cols:
        # Shift target by -1 to predict T+1
        df_shifted = df.copy()
        df_shifted['Target'] = df_shifted[target].shift(-1)
        
        # Drop the last row which will be NaN due to shift(-1)
        df_shifted.dropna(subset=['Target'], inplace=True)
        
        feature_cols = [c for c in df.columns]
        X = df_shifted[feature_cols]
        y = df_shifted['Target']
        
        # Chronological train-test split (80/20)
        split_idx = int(len(df_shifted) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Native Feature Importance
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        results[target] = {
            'model': rf,
            'feature_names': feature_cols,
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            },
            'feature_importances': feature_importance_df
        }
        
    return results
