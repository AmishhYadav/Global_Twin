import os
import sys

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.test_ingest import generate_mock_data
from src.data.ingest import load_and_clean_data
from src.features.build_features import create_time_series_features
from src.models.train import train_models

def run_ml_pipeline_test():
    test_file = "test_ml_data.csv"
    try:
        # Generate enough data to survive our 14-day rolling windows and 80/20 splits
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        data = {
            'Date': dates,
            'Oil_Price': np.random.normal(60, 5, len(dates)),
            'Shipping_Cost': np.random.normal(1500, 200, len(dates)),
            'Inflation_Rate': np.random.normal(2.5, 0.5, len(dates))
        }
        # Introduce a trend to test basic R2
        data['Oil_Price'] += np.linspace(0, 20, len(dates))
        data['Shipping_Cost'] += np.linspace(0, 500, len(dates))
        
        df = pd.DataFrame(data)
        df.to_csv(test_file, index=False)
        print(f"Created mock data at {test_file}")
        
        print("\nPipeline Stage 1: Ingestion")
        clean_df = load_and_clean_data(test_file)
        
        print("\nPipeline Stage 2: Feature Engineering")
        feat_df = create_time_series_features(clean_df, lags=[1, 3, 7], rolling_windows=[7, 14])
        print(f"Features Generated. Shape: {feat_df.shape}")
        
        print("\nPipeline Stage 3: Training Models")
        target_cols = ['Oil_Price', 'Shipping_Cost']
        results = train_models(feat_df, target_cols)
        
        print("\n--- RESULTS ---")
        for target, output in results.items():
            print(f"\nTarget: {target}")
            print(f"  RMSE: {output['metrics']['RMSE']:.4f}")
            print(f"  MAE:  {output['metrics']['MAE']:.4f}")
            print(f"  R2:   {output['metrics']['R2']:.4f}")
            
            print("  Top 3 Features:")
            print(output['feature_importances'].head(3).to_string(index=False))
            
        print("\nSUCCESS: ML training pipeline executed perfectly.")
        sys.exit(0)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    run_ml_pipeline_test()
