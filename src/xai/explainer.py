import json
import networkx as nx

def generate_xai_payload(sim_dict, models_dict, G, shock_node):
    """
    Generates JSON payload wrapping predictions with static ±RMSE bounds and DAG narratives.
    
    Args:
        sim_dict: Output from run_simulation (baseline and shocked trajectories)
        models_dict: Output from train_models (contains RMSE)
        G: NetworkX DAG generated in Phase 3
        shock_node: Origin coordinate of the user's shock
        
    Returns:
        JSON string mapping timelines, bounding ranges, and text causality explanations.
    """
    payload = {
        "shock_origin": shock_node,
        "nodes": {}
    }
    
    baseline_steps = sim_dict['baseline']
    shocked_steps = sim_dict['shocked']
    horizon = len(baseline_steps)
    
    # Target elements natively modeled
    targets = models_dict.keys()
    
    for target in targets:
        node_data = {
            "trajectory": [],
            "explanation": ""
        }
        
        # 1. Bounds & Trajectory Mapping
        rmse = models_dict[target]['metrics']['RMSE']
        for step in range(horizon):
            b_val = baseline_steps[step][target]
            s_val = shocked_steps[step][target]
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
                    "margin_of_error_rmse": rmse
                }
            })
            
        # 2. XAI Narrative via DAG trace translation
        if target == shock_node:
            node_data["explanation"] = f"Target variable {target} was structurally shocked by user."
        elif nx.has_path(G, source=shock_node, target=target):
            # Calculate the explicit structural pathway
            path = nx.shortest_path(G, source=shock_node, target=target)
            path_str = " -> ".join(path)
            node_data["explanation"] = f"Causal Impact Path: [{path_str}]. Trajectory deviations strictly driven chronologically across these graph dependencies based on historical Random Forest logic."
        else:
            node_data["explanation"] = f"No mathematical DAG topological connection found linking [{shock_node}] to [{target}]."
            
        payload["nodes"][target] = node_data
        
    return json.dumps(payload, indent=4)
