"""
Global Twin — XAI Explainer (v2.0)

Generates human-readable explanations of WHY each variable changed,
with real-world cause-effect relationships in plain English.
"""

import json
import networkx as nx


# ── Plain English explanations for economic relationships ──
RELATIONSHIP_EXPLANATIONS = {
    ("CRUDE_OIL", "GOLD"): "When oil rises, investors buy gold as a hedge against inflation",
    ("CRUDE_OIL", "SP500"): "Higher oil = higher costs for companies = lower profits = stocks fall",
    ("CRUDE_OIL", "US_CPI_INFLATION"): "Oil drives transport/manufacturing costs, pushing consumer prices up",
    ("CRUDE_OIL", "EU_CPI_INFLATION"): "Europe imports most of its oil, so price spikes hit inflation hard",
    ("CRUDE_OIL", "IN_CPI_INFLATION"): "India imports 85% of its oil — price spikes directly raise inflation",
    ("CRUDE_OIL", "INR_USD"): "India pays for oil in USD — costlier oil means more dollars needed, weakening the Rupee",
    ("CRUDE_OIL", "BALTIC_DRY_INDEX"): "Shipping fuel costs rise with oil, driving up shipping rates",
    ("CRUDE_OIL", "NATURAL_GAS"): "Oil and gas are substitute energy sources — they tend to move together",
    ("US_FED_RATE", "INR_USD"): "Higher US rates attract money to USD — Rupee weakens as dollars flow out of India",
    ("US_FED_RATE", "EUR_USD"): "Higher US rates make USD more attractive vs Euro",
    ("US_FED_RATE", "SP500"): "Higher rates = companies borrow less = slower growth = stocks fall",
    ("US_FED_RATE", "GOLD"): "Higher rates make bonds more attractive than gold (gold pays no interest)",
    ("VIX", "SP500"): "VIX measures fear — when fear spikes, investors sell stocks",
    ("VIX", "GOLD"): "High fear drives investors to safe-haven assets like gold",
    ("SP500", "GOLD"): "When stocks crash, investors flee to gold for safety",
    ("US_GDP_GROWTH", "SP500"): "Stronger economy = higher corporate earnings = stocks rise",
    ("US_GDP_GROWTH", "US_CPI_INFLATION"): "Fast growth can overheat the economy, pushing prices up",
    ("CN_GDP_GROWTH", "CRUDE_OIL"): "China is the world's largest oil importer — its growth drives global oil demand",
    ("CN_GDP_GROWTH", "COPPER"): "China uses ~50% of world's copper (construction/electronics) — growth = more demand",
    ("GOLD", "INR_USD"): "India is the world's 2nd largest gold consumer — gold imports affect the Rupee",
    ("INR_USD", "IN_GDP_GROWTH"): "A weaker Rupee makes imports expensive, slowing economic growth",
}

# ── Variable descriptions ──
VAR_DESCRIPTIONS = {
    "CRUDE_OIL": "Oil price (per barrel)",
    "SP500": "US stock market (S&P 500)",
    "GOLD": "Gold price (per ounce)",
    "VIX": "Market fear index",
    "US_GDP_GROWTH": "US economic growth",
    "US_CPI_INFLATION": "US consumer prices",
    "US_UNEMPLOYMENT": "US jobless rate",
    "EU_GDP_GROWTH": "EU economic growth",
    "EU_CPI_INFLATION": "EU consumer prices",
    "CN_GDP_GROWTH": "China's economic growth",
    "IN_GDP_GROWTH": "India's economic growth",
    "JP_GDP_GROWTH": "Japan's economic growth",
    "EUR_USD": "Euro/Dollar rate",
    "CNY_USD": "Yuan/Dollar rate",
    "INR_USD": "Rupee/Dollar rate",
    "JPY_USD": "Yen/Dollar rate",
    "NATURAL_GAS": "Natural gas price",
    "COPPER": "Copper price",
    "WHEAT": "Wheat price",
    "BALTIC_DRY_INDEX": "Global shipping cost index",
    "SEMICONDUCTOR_IDX": "Semiconductor stocks index",
    "US_FED_RATE": "US interest rate",
}


def _get_var_name(var):
    return VAR_DESCRIPTIONS.get(var, var)


def _get_relationship(source, target):
    """Get plain English explanation of why source affects target."""
    # Direct lookup
    key = (source, target)
    if key in RELATIONSHIP_EXPLANATIONS:
        return RELATIONSHIP_EXPLANATIONS[key]
    # Reverse (effect still valid)
    rev = (target, source)
    if rev in RELATIONSHIP_EXPLANATIONS:
        return RELATIONSHIP_EXPLANATIONS[rev]
    return None


def generate_xai_payload(sim_dict, models_dict, G, shock_node=None, shocks=None):
    """
    Generate rich XAI payload with plain-English cause-effect explanations.
    """
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
        "chain_reaction": [],  # New: ordered cause-effect chain
        "nodes": {},
    }

    baseline_steps = sim_dict['baseline']
    shocked_steps = sim_dict['shocked']
    horizon = len(baseline_steps)

    # Build chain reaction summary
    chain = []
    for origin in shock_origins:
        pct = shocks.get(origin, 0)
        chain.append({
            "variable": origin,
            "name": _get_var_name(origin),
            "change": f"{pct*100:+.0f}%",
            "type": "direct_shock",
            "reason": f"You changed this variable by {pct*100:+.0f}%",
        })

    for target in models_dict.keys():
        if target in shock_origins:
            continue

        # Find the best explanation from any shock origin
        best_explanation = None
        best_path = None

        for origin in shock_origins:
            if origin not in G.nodes or target not in G.nodes:
                continue
            if nx.has_path(G, source=origin, target=target):
                path = nx.shortest_path(G, source=origin, target=target)
                # Look for relationship explanation along the path
                for i in range(len(path) - 1):
                    explanation = _get_relationship(path[i], path[i+1])
                    if explanation:
                        best_explanation = explanation
                        best_path = path
                        break
                if best_explanation:
                    break

        # Calculate impact
        if horizon > 0 and target in baseline_steps[0] and target in shocked_steps[0]:
            final_b = baseline_steps[-1].get(target, 0)
            final_s = shocked_steps[-1].get(target, 0)
            if final_b != 0:
                impact_pct = ((final_s - final_b) / abs(final_b)) * 100
            else:
                impact_pct = 0

            if abs(impact_pct) > 0.01:  # Only show variables that actually changed
                chain.append({
                    "variable": target,
                    "name": _get_var_name(target),
                    "change": f"{impact_pct:+.1f}%",
                    "type": "cascade",
                    "reason": best_explanation or f"Correlated with shocked variables through ML patterns",
                    "path": " → ".join(best_path) if best_path else None,
                })

    payload["chain_reaction"] = chain

    # Per-variable detailed data
    for target in models_dict.keys():
        node_data = {
            "trajectory": [],
            "explanation": "",
            "is_shock_origin": target in shock_origins,
            "var_name": _get_var_name(target),
        }

        rmse = models_dict[target]['metrics']['RMSE']
        model_name = models_dict[target].get('model_name', 'Random Forest')

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

        # Build explanation
        node_data["explanation"] = _build_narrative(target, shock_origins, shocks, G)
        payload["nodes"][target] = node_data

    return json.dumps(payload, indent=2)


def _build_narrative(target, shock_origins, shocks, G):
    """Build plain-English causal narrative."""
    if target in shock_origins:
        pct = shocks.get(target, 0)
        return f"⚡ You directly changed {_get_var_name(target)} by {pct*100:+.0f}%"

    explanations = []
    for origin in shock_origins:
        if origin not in G.nodes or target not in G.nodes:
            continue
        if nx.has_path(G, source=origin, target=target):
            path = nx.shortest_path(G, source=origin, target=target)
            pct = shocks.get(origin, 0)

            # Find human explanation for each hop
            hop_explanations = []
            for i in range(len(path) - 1):
                rel = _get_relationship(path[i], path[i+1])
                if rel:
                    hop_explanations.append(rel)

            path_names = [_get_var_name(p) for p in path]
            chain_str = " → ".join(path_names)

            if hop_explanations:
                explanations.append(f"📍 {chain_str}: {hop_explanations[0]}")
            else:
                explanations.append(f"📍 {chain_str}: ML detected a predictive pattern")

    if explanations:
        return "\n".join(explanations)

    return f"📊 No direct causal path found, but ML detected indirect correlation patterns"
