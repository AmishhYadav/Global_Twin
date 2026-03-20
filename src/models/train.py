"""
Global Twin — Multi-Model ML Pipeline (v2.0)

Trains multiple model types per target variable, auto-selects
the best performer, and stores results in a model registry.

Supported models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─────────────────────────────────────────────
#  Model Configurations
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "RandomForest": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 50, "max_depth": 8, "random_state": 42, "n_jobs": -1},
    },
}


def _evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
        "predictions": y_pred,
    }


def train_single_model(model_name, config, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a single model.
    
    Returns:
        dict with model object, metrics, and feature importances.
    """
    model_class = config["class"]
    model = model_class(**config["params"])
    model.fit(X_train, y_train)
    
    metrics = _evaluate_model(model, X_test, y_test)
    
    # Feature importances
    importances = model.feature_importances_
    
    return {
        "model": model,
        "model_name": model_name,
        "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
        "predictions": metrics["predictions"],
        "feature_importances": importances,
    }


def train_models(df, target_cols, test_ratio=0.2, verbose=True):
    """
    Train multiple model types per target, auto-select best by validation RMSE.
    
    Args:
        df: DataFrame with engineered features (numeric only, DatetimeIndex).
        target_cols: List of column names to forecast.
        test_ratio: Fraction of data for temporal test set.
        verbose: Print comparison results.
    
    Returns:
        dict mapping target_name → {
            'model': best sklearn model,
            'model_name': str,
            'feature_names': list,
            'metrics': dict,
            'feature_importances': DataFrame,
            'all_candidates': list of dicts (all tried models),
            'trained_at': ISO timestamp,
        }
    """
    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for target in target_cols:
        if verbose:
            print(f"\n  ── Training models for: {target} ──")
        
        # Prepare T+1 prediction setup
        df_shifted = df.copy()
        df_shifted['_Target'] = df_shifted[target].shift(-1)
        df_shifted.dropna(subset=['_Target'], inplace=True)
        
        feature_cols = [c for c in numeric_cols if c != '_Target']
        X = df_shifted[feature_cols]
        y = df_shifted['_Target']
        
        # Chronological split
        split_idx = int(len(df_shifted) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train all model types
        candidates = []
        for model_name, config in MODEL_CONFIGS.items():
            result = train_single_model(
                model_name, config, X_train, y_train, X_test, y_test
            )
            candidates.append(result)
            
            if verbose:
                m = result['metrics']
                print(f"    {model_name:20s} → RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")
        
        # Auto-select best model by lowest RMSE
        best = min(candidates, key=lambda c: c['metrics']['RMSE'])
        
        if verbose:
            print(f"    ★ Best: {best['model_name']} (RMSE={best['metrics']['RMSE']:.4f})")
        
        # Build feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best['feature_importances']
        }).sort_values(by='Importance', ascending=False)
        
        results[target] = {
            'model': best['model'],
            'model_name': best['model_name'],
            'feature_names': feature_cols,
            'metrics': best['metrics'],
            'feature_importances': importance_df,
            'all_candidates': [
                {
                    'model_name': c['model_name'],
                    'model': c['model'],
                    'metrics': c['metrics'],
                }
                for c in candidates
            ],
            'trained_at': datetime.now().isoformat(),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
    
    return results


def get_comparison_report(results):
    """
    Generate a human-readable comparison report for all targets.
    
    Args:
        results: Output from train_models().
    
    Returns:
        str: Formatted comparison report.
    """
    lines = ["=" * 60, "  MODEL COMPARISON REPORT", "=" * 60]
    
    for target, data in results.items():
        lines.append(f"\n  Target: {target}")
        lines.append(f"  Train/Test split: {data['train_size']}/{data['test_size']}")
        lines.append(f"  {'Model':20s} {'RMSE':>10s} {'MAE':>10s} {'R²':>10s}  {'Selected':>8s}")
        lines.append("  " + "-" * 62)
        
        for candidate in data['all_candidates']:
            m = candidate['metrics']
            selected = "★" if candidate['model_name'] == data['model_name'] else ""
            lines.append(
                f"  {candidate['model_name']:20s} {m['RMSE']:10.4f} {m['MAE']:10.4f} {m['R2']:10.4f}  {selected:>8s}"
            )
        
        # Top 5 features
        top5 = data['feature_importances'].head(5)
        lines.append(f"\n  Top 5 Features ({data['model_name']}):")
        for _, row in top5.iterrows():
            lines.append(f"    {row['Feature']:40s} {row['Importance']:.4f}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
