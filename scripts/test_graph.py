import os
import sys
import json
import pandas as pd
import numpy as np
import networkx as nx

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingest import load_and_clean_data
from src.features.build_features import create_time_series_features
from src.models.train import train_models
from src.graph.build_graph import create_knowledge_graph

def run_graph_pipeline_test():
    test_file = "test_graph_data.csv"
    graph_path = "models/test_graph.json"
    
    try:
        # Generate correlated mock data so we get a realistic graph
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        
        # Base independent variable
        oil = np.random.normal(60, 5, len(dates))
        
        # Dependent variables with intentional correlations
        # Shipping depends heavily on Oil
        shipping = (oil * 5) + np.random.normal(100, 50, len(dates))
        
        # Inflation depends on both but delayed (meaning temporal model will catch it)
        inflation = (shipping * 0.01) + np.random.normal(2.0, 0.1, len(dates))
        
        data = {
            'Date': dates,
            'Oil_Price': oil,
            'Shipping_Cost': shipping,
            'Inflation_Rate': inflation
        }
        df = pd.DataFrame(data)
        df.to_csv(test_file, index=False)
        print(f"Created correlated mock data at {test_file}")
        
        # Stages 1 & 2
        clean_df = load_and_clean_data(test_file)
        feat_df = create_time_series_features(clean_df, lags=[1, 3], rolling_windows=[7])
        
        print("\nTraining ML Models...")
        target_cols = ['Shipping_Cost', 'Inflation_Rate']
        results = train_models(feat_df, target_cols)
        
        # Stage 3: Graph Build
        print("\nPipeline Stage 3: Building Knowledge Graph")
        G = create_knowledge_graph(results, importance_threshold=0.05, save_path=graph_path)
        
        print("\n--- GRAPH VALIDATION ---")
        assert nx.is_directed_acyclic_graph(G), "Graph is not a DAG!"
        print(f"Graph is a strict DAG: {nx.is_directed_acyclic_graph(G)}")
        print(f"Nodes: {G.nodes()}")
        print(f"Edges: {G.edges(data=True)}")
        
        assert os.path.exists(graph_path), "JSON Artifact was not saved."
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
            assert 'nodes' in graph_data and 'links' in graph_data
            print("JSON Node Link Object schema verified.")
            
        print("\nSUCCESS: Knowledge Graph generated perfectly.")
        sys.exit(0)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(graph_path):
            os.remove(graph_path)

if __name__ == "__main__":
    run_graph_pipeline_test()
