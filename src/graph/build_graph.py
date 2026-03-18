import networkx as nx
import json
import os

def extract_base_variable(feature_name):
    """Clean feature string back to its core economic variable name."""
    if '_rolling' in feature_name:
        return feature_name.split('_rolling')[0]
    elif '_lag' in feature_name:
        return feature_name.split('_lag')[0]
    elif '_pct_change' in feature_name:
        return feature_name.split('_pct_change')[0]
    return feature_name

def create_knowledge_graph(ml_results_dict, importance_threshold=0.05, save_path="models/graph.json"):
    G = nx.DiGraph()
    
    # Aggregate importance by base variable
    aggregated_edges = {} # (source, target) -> total_importance
    
    for target_var, result in ml_results_dict.items():
        importance_df = result['feature_importances']
        
        for _, row in importance_df.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            source_var = extract_base_variable(feature)
            
            # Avoid self-loops (e.g. past Oil predicting future Oil, though valid, we want a causal graph between DIFFERENT variables)
            if source_var == target_var:
                continue
                
            if importance > importance_threshold:
                edge_tuple = (source_var, target_var)
                aggregated_edges[edge_tuple] = aggregated_edges.get(edge_tuple, 0) + importance
                
    # Add nodes and edges
    for (source, target), weight in aggregated_edges.items():
        G.add_edge(source, target, weight=weight)
        
    # Enforce DAG
    while not nx.is_directed_acyclic_graph(G):
        # find elementary circuits
        try:
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                break
                
            # Find the cycle with the weakest link to break
            cycle = cycles[0]
            # cycle is a list of nodes: [A, B, C] meaning A->B, B->C, C->A
            weakest_edge = None
            min_weight = float('inf')
            
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    weight = G[u][v]['weight']
                    if weight < min_weight:
                        min_weight = weight
                        weakest_edge = (u, v)
                    
            if weakest_edge:
                print(f"Cycle detected: {cycle}. Breaking weakest causal edge {weakest_edge} (Weight: {min_weight:.4f})")
                G.remove_edge(*weakest_edge)
        except nx.NetworkXNoCycle:
            break
            
    # Export
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(nx.node_link_data(G), f, indent=4)
        print(f"Graph dynamically saved to {save_path}")
        
    return G
