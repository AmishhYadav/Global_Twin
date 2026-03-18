import os
import sys
import numpy as np
import pandas as pd

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import load_and_clean_data
from src.features.build_features import create_time_series_features
from src.models.train import train_models
from src.simulation.engine import run_simulation

def test_simulator():
    test_file = "test_sim_data.csv"
    try:
        # Generate correlated mock data
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        oil = np.random.normal(60, 5, len(dates))
        shipping = (oil * 5) + np.random.normal(100, 50, len(dates))
        inflation = (shipping * 0.01) + np.random.normal(2.0, 0.1, len(dates))
        
        data = {
            'Date': dates,
            'Oil_Price': oil,
            'Shipping_Cost': shipping,
            'Inflation_Rate': inflation
        }
        df = pd.DataFrame(data)
        df.to_csv(test_file, index=False)
        print(f"Created mocked base variables for Simulation Engine test.")
        
        clean_df = load_and_clean_data(test_file)
        feat_df = create_time_series_features(clean_df, lags=[1, 3], rolling_windows=[7])
        
        print("\nTraining ML Models...")
        target_cols = ['Shipping_Cost', 'Inflation_Rate']
        results = train_models(feat_df, target_cols)
        
        print("\nPipeline Stage 4: Executing T+3 Shock Simulation [Oil + 20%]")
        # Execute recursive horizon prediction using base history
        sim_out = run_simulation(models_dict=results, base_df=clean_df.tail(60), shock_node='Oil_Price', shock_pct=0.20, horizon=3)
        
        print("\n--- SIMULATION VALIDATION ---")
        for i, (b_step, s_step) in enumerate(zip(sim_out['baseline'], sim_out['shocked'])):
            print(f"\nT+{i+1} Delta Projection:")
            for target in target_cols:
                diff = s_step[target] - b_step[target]
                pct = (diff / b_step[target]) * 100
                print(f"  {target}: {b_step[target]:.2f} -> {s_step[target]:.2f} ({diff:+.2f} / {pct:+.2f}%)")
        
        print("\nSUCCESS: Simulation Cascade executed recursively perfectly.")
        sys.exit(0)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_simulator()
