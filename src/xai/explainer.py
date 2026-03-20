"""
Global Twin — XAI Explainer (v2.0)

Generates structured JSON payloads with:
  - Per-variable trajectory (baseline vs shocked)
  - ±RMSE confidence bounds
  - DAG-traced causal narratives
  - Multi-shock scenario context
"""

import json
import networkx as nx


def generate_xai_payload(sim_dict, models_dict, G, shock_node=None, shocks=None):
    """
    Generate XAI payload for simulation results.
    
    Args:
        sim_dict: Output from run_simulation.
        models_dict: Output from train_models (contains RMSE).
        G: NetworkX DAG.
        shock_node: (v1 compat) Single shock origin.
        shocks: dict of all shocked variables (for v2 multi-shock).
    
    Returns:
        JSON string with trajectories, bounds, and explanations.
    """
    # Build shock origins list
    if shocks is None:
        applied = sim_dict.get('applied_shocks', {})
        if applied:
            shocks = {k: v['pct'] for k, v in applied.items()}
        elif shock_node:
            shocks = {shock_node: 0.0}
        else:
            shocks = {}
    
    shock_origins = list(shocks.keys())
    
    payload = {
        "scenario_name": sim_dict.get('scenario_name', 'Custom Shock'),
        "shock_origins": shock_origins,
        "applied_shocks": sim_dict.get('applied_shocks', {}),
        "horizon": sim_dict.get('horizon', len(sim_dict['baseline'])),
        "nodes": {},
    }
    
    baseline_steps = sim_dict['baseline']
    shocked_steps = sim_dict['shocked']
    horizon = len(baseline_steps)
    
    for target in models_dict.keys():
        node_data = {
            "trajectory": [],
            "explanation": "",
            "is_shock_origin": target in shock_origins,
        }
        
        rmse = models_dict[target]['metrics']['RMSE']
        model_name = models_dict[target].get('model_name', 'Unknown')
        
        for step in range(horizon):
            b_val = baseline_steps[step].get(target, 0)
            s_val = shocked_steps[step].get(target, 0)
            delta = s_val - b_val
            pct_delta = (delta / b_val) * 100 if b_val != 0 else 0
            
            node_data["trajectory"].append({
                "step": step + 1,
                "baseline": b_val,
                "shocked": s_val,
                "absolute_delta": delta,
                "percentage_delta": pct_delta,
                "bounds": {
                    "lower_bound": s_val - rmse,
                    "upper_bound": s_val + rmse,
                    "rmse": rmse,
                },
            })
        
        # XAI Narrative
        node_data["explanation"] = _build_narrative(
            target, shock_origins, shocks, G, model_name
        )
        
        payload["nodes"][target] = node_data
    
    return json.dumps(payload, indent=2)


def _build_narrative(target, shock_origins, shocks, G, model_name):
    """Build a human-readable causal narrative for a target variable."""
    
    # Direct shock
    if target in shock_origins:
        pct = shocks.get(target, 0)
        return (
            f"{target} was directly shocked by {pct*100:+.0f}%. "
            f"Predictions modeled via {model_name}."
        )
    
    # Find causal paths from any shock origin
    paths = []
    for origin in shock_origins:
        if origin not in G.nodes or target not in G.nodes:
            continue
        if nx.has_path(G, source=origin, target=target):
            path = nx.shortest_path(G, source=origin, target=target)
            pct = shocks.get(origin, 0)
            paths.append((origin, pct, path))
    
    if paths:
        narratives = []
        for origin, pct, path in paths:
            path_str = " → ".join(path)
            narratives.append(
                f"{origin} ({pct*100:+.0f}%) cascades via [{path_str}]"
            )
        return (
            f"Impacted through {len(paths)} causal path(s): " +
            "; ".join(narratives) +
            f". Predicted by {model_name}."
        )
    
    return f"No direct DAG path from shocked variables to {target}. Indirect effects may exist via correlated features ({model_name})."
