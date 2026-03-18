import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import load_and_clean_data
from src.features.build_features import create_time_series_features
from src.models.train import train_models
from src.graph.build_graph import create_knowledge_graph
from src.simulation.engine import run_simulation
from src.xai.explainer import generate_xai_payload

def run_xai_pipeline_test():
    test_file = "test_xai_data.csv"
    graph_path = "models/test_xai_graph.json"
    try:
        # Generate correlated mock data so we get a realistic graph connecting Oil to Inflation
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
        print(f"Created correlated mock cascade data.")
        
        # Pipeline execution Stages 1-4 natively
        clean_df = load_and_clean_data(test_file)
        feat_df = create_time_series_features(clean_df, lags=[1, 3], rolling_windows=[7])
        
        target_cols = ['Shipping_Cost', 'Inflation_Rate']
        results = train_models(feat_df, target_cols)
        
        G = create_knowledge_graph(results, importance_threshold=0.05, save_path=graph_path)
        
        shock = 0.20
        sim_out = run_simulation(models_dict=results, base_df=clean_df.tail(60), shock_node='Oil_Price', shock_pct=shock, horizon=3)
        
        print("\nPipeline Stage 5: Generating XAI JSON Payload")
        payload_str = generate_xai_payload(sim_out, results, G, 'Oil_Price')
        
        print("\n--- XAI PAYLOAD FORMATTING VALIDATION ---")
        print(payload_str)
        
        # Self-Check assertions
        parsed = json.loads(payload_str)
        assert 'nodes' in parsed
        assert 'Shipping_Cost' in parsed['nodes']
        assert 'bounds' in parsed['nodes']['Shipping_Cost']['trajectory'][0]
        assert 'margin_of_error_rmse' in parsed['nodes']['Shipping_Cost']['trajectory'][0]['bounds']
        assert 'Oil_Price ->' in parsed['nodes']['Shipping_Cost']['explanation']
        
        print("\nSUCCESS: XAI Explainability formatter executed perfectly.")
        sys.exit(0)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(graph_path):
            os.remove(graph_path)

if __name__ == "__main__":
    run_xai_pipeline_test()
