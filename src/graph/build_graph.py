"""
Global Twin — Cross-Country Knowledge Graph (v2.0)

Builds a massive Directed Acyclic Graph spanning all countries
with both intra-country and cross-border causal edges.

Node types:
  - Country-specific indicators (e.g., US_UNEMPLOYMENT)
  - Global indicators (e.g., CRUDE_OIL)
  - Composite indicators (e.g., GLOBAL_GDP_INDEX)

Edge types:
  - ML-derived: from feature importance (variable A predicts B)
  - Structural: hard-coded economic relationships
"""

import networkx as nx
import json
import os
from src.data.indicators import COUNTRIES, COUNTRY_INDICATORS, GLOBAL_INDICATORS


# ─────────────────────────────────────────────
#  Feature Name → Base Variable Extraction
# ─────────────────────────────────────────────

def extract_base_variable(feature_name):
    """Clean engineered feature name back to its core indicator name."""
    # Remove v2.0 feature engineering suffixes
    suffixes = [
        '_rmean_', '_rstd_', '_roc_', '_momentum_', '_lag_',
        '_rolling_mean_', '_rolling_std_', '_pct_change_',
    ]
    for suffix in suffixes:
        if suffix in feature_name:
            return feature_name.split(suffix)[0]
    
    # Cross-country features (xc_*) → extract the driver indicator
    if feature_name.startswith('xc_'):
        # e.g., xc_USrate_to_INR_lag3 → US_FED_RATE (mapped below)
        return None  # Skip — these are composite, handled by structural edges
    
    # Sector features
    if feature_name.startswith('sector_'):
        return None  # Skip — handled by structural edges
    
    # Global composites
    if feature_name.startswith('GLOBAL_') or feature_name in ('COMMODITY_PRESSURE', 'USD_STRENGTH'):
        return feature_name
    
    return feature_name


# ─────────────────────────────────────────────
#  Structural (Hard-Coded) Cross-Border Edges
# ─────────────────────────────────────────────

STRUCTURAL_EDGES = [
    # Cross-border: US monetary policy → exchange rates
    ("US_FED_RATE", "EUR_USD", 0.6),
    ("US_FED_RATE", "CNY_USD", 0.5),
    ("US_FED_RATE", "INR_USD", 0.7),
    ("US_FED_RATE", "JPY_USD", 0.6),
    
    # Cross-border: EU monetary policy → EUR/USD
    ("EU_ECB_RATE", "EUR_USD", 0.5),
    
    # Cross-border: Japan rate → JPY
    ("JP_BOJ_RATE", "JPY_USD", 0.5),
    
    # Commodity → Inflation linkages (cross-border)
    ("CRUDE_OIL", "US_CPI_INFLATION", 0.6),
    ("CRUDE_OIL", "EU_CPI_INFLATION", 0.5),
    ("CRUDE_OIL", "IN_CPI_INFLATION", 0.7),
    ("CRUDE_OIL", "JP_CPI_INFLATION", 0.4),
    ("CRUDE_OIL", "CN_CPI_INFLATION", 0.4),
    
    # Oil → Shipping
    ("CRUDE_OIL", "BALTIC_DRY_INDEX", 0.5),
    
    # Energy sector linkages
    ("CRUDE_OIL", "NATURAL_GAS", 0.3),
    
    # Fear/Risk → Markets
    ("VIX", "SP500", 0.7),
    ("VIX", "GOLD", 0.4),
    
    # Tech → Markets
    ("SEMICONDUCTOR_IDX", "SP500", 0.4),
    
    # US macro → global
    ("US_GDP_GROWTH", "SP500", 0.5),
    ("US_UNEMPLOYMENT", "US_CONSUMER_CONFIDENCE", 0.6),
    ("US_CONSUMER_CONFIDENCE", "US_GDP_GROWTH", 0.3),
    
    # Trade linkages
    ("CN_GDP_GROWTH", "US_TRADE_BALANCE", 0.4),
    ("CRUDE_OIL", "US_TRADE_BALANCE", 0.3),
    
    # Exchange rate → trade
    ("INR_USD", "IN_GDP_GROWTH", 0.3),
    ("CNY_USD", "CN_GDP_GROWTH", 0.3),
    ("JPY_USD", "JP_GDP_GROWTH", 0.3),
    ("EUR_USD", "EU_GDP_GROWTH", 0.3),
    
    # Commodity pressure flows
    ("WHEAT", "IN_CPI_INFLATION", 0.4),
    ("WHEAT", "CN_CPI_INFLATION", 0.3),
    ("COPPER", "CN_GDP_GROWTH", 0.3),
    ("GOLD", "INR_USD", 0.2),
]


# ─────────────────────────────────────────────
#  Node Metadata
# ─────────────────────────────────────────────

def _get_node_metadata(indicator_name):
    """Assign country, sector, and type metadata to a node."""
    # Determine country
    country = "GLOBAL"
    for code, indicators in COUNTRY_INDICATORS.items():
        if indicator_name in indicators:
            country = code
            break
    
    # Determine sector
    sector = "macro"
    if any(kw in indicator_name for kw in ['OIL', 'GAS', 'GOLD', 'COPPER', 'WHEAT']):
        sector = "commodity"
    elif any(kw in indicator_name for kw in ['EUR_USD', 'CNY_USD', 'INR_USD', 'JPY_USD', 'USD_STRENGTH']):
        sector = "forex"
    elif any(kw in indicator_name for kw in ['SP500', 'VIX', 'SEMICONDUCTOR', 'BALTIC']):
        sector = "market"
    elif 'GDP' in indicator_name:
        sector = "gdp"
    elif 'CPI' in indicator_name or 'INFLATION' in indicator_name:
        sector = "inflation"
    elif 'RATE' in indicator_name or 'FED' in indicator_name or 'ECB' in indicator_name or 'BOJ' in indicator_name:
        sector = "monetary"
    elif 'UNEMPLOYMENT' in indicator_name:
        sector = "labor"
    
    return {"country": country, "sector": sector}


# ─────────────────────────────────────────────
#  Main Graph Builder
# ─────────────────────────────────────────────

def create_knowledge_graph(ml_results_dict, importance_threshold=0.03, save_path="models/graph.json"):
    """
    Build a cross-country knowledge graph from ML feature importances
    plus structural economic relationships.
    
    Args:
        ml_results_dict: Output from train_models().
        importance_threshold: Minimum feature importance to create an edge.
        save_path: Path to save JSON graph artifact.
    
    Returns:
        nx.DiGraph with node/edge metadata.
    """
    G = nx.DiGraph()
    
    # ── 1. ML-Derived Edges ──
    aggregated_edges = {}
    
    for target_var, result in ml_results_dict.items():
        importance_df = result['feature_importances']
        
        for _, row in importance_df.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            source_var = extract_base_variable(feature)
            
            if source_var is None:
                continue
            if source_var == target_var:
                continue
            if importance < importance_threshold:
                continue
            
            edge_tuple = (source_var, target_var)
            aggregated_edges[edge_tuple] = aggregated_edges.get(edge_tuple, 0) + importance
    
    for (source, target), weight in aggregated_edges.items():
        G.add_edge(source, target, weight=round(weight, 4), edge_type="ml_derived")
    
    ml_edge_count = G.number_of_edges()
    
    # ── 2. Structural Edges ──
    structural_count = 0
    for source, target, weight in STRUCTURAL_EDGES:
        if not G.has_edge(source, target):
            G.add_edge(source, target, weight=weight, edge_type="structural")
            structural_count += 1
        else:
            # Boost existing ML edge with structural knowledge
            G[source][target]['weight'] = round(
                G[source][target]['weight'] + weight * 0.3, 4
            )
            G[source][target]['edge_type'] = "hybrid"
    
    # ── 3. Node Metadata ──
    for node in G.nodes():
        meta = _get_node_metadata(node)
        G.nodes[node].update(meta)
    
    # ── 4. Enforce DAG ──
    cycles_broken = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                break
            
            cycle = cycles[0]
            weakest_edge = None
            min_weight = float('inf')
            
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    w = G[u][v]['weight']
                    if w < min_weight:
                        min_weight = w
                        weakest_edge = (u, v)
            
            if weakest_edge:
                G.remove_edge(*weakest_edge)
                cycles_broken += 1
        except nx.NetworkXNoCycle:
            break
    
    # ── 5. Export ──
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(nx.node_link_data(G), f, indent=2)
    
    # Summary
    countries_in_graph = set(nx.get_node_attributes(G, 'country').values())
    sectors_in_graph = set(nx.get_node_attributes(G, 'sector').values())
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Edge breakdown: {ml_edge_count} ML-derived, {structural_count} structural, {cycles_broken} cycles broken")
    print(f"  Countries: {sorted(countries_in_graph)}")
    print(f"  Sectors: {sorted(sectors_in_graph)}")
    if save_path:
        print(f"  Saved to {save_path}")
    
    return G


def get_graph_summary(G):
    """Generate a structured summary of the knowledge graph."""
    countries = nx.get_node_attributes(G, 'country')
    sectors = nx.get_node_attributes(G, 'sector')
    
    summary = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(G),
        "countries": {},
        "sectors": {},
        "cross_border_edges": 0,
    }
    
    # Count by country
    for node, country in countries.items():
        summary["countries"].setdefault(country, []).append(node)
    
    # Count by sector
    for node, sector in sectors.items():
        summary["sectors"].setdefault(sector, []).append(node)
    
    # Count cross-border edges
    for u, v in G.edges():
        u_country = countries.get(u, "GLOBAL")
        v_country = countries.get(v, "GLOBAL")
        if u_country != v_country:
            summary["cross_border_edges"] += 1
    
    return summary
