---
wave: 1
depends_on: []
files_modified: ["src/xai/explainer.py", "scripts/test_xai.py"]
autonomous: true
---

# Plan 1: XAI Narrative Formatter

## Objective
Implement an XAI wrapper that intercepts the Simulation Engine's output arrays, calculates ±RMSE ranges, constructs a recursive DAG dependency trace description, and outputs a unified JSON string.

## Tasks

<task>
<action>
Create directory `src/xai/` and build `src/xai/explainer.py`. 
Implement `generate_xai_payload(sim_dict, models_dict, G, shock_node)`.
Logic:
1. Initialize an empty nested dictionary matching Target Node Keys.
2. Iterate all targets in `sim_dict['shocked']` across the simulated Steps mapping the Absolute Delta.
3. Attach confidence intervals: Extract `RMSE` from `models_dict[target]['metrics']`. Add nested dictionaries `upper_bound: Val + RMSE` and `lower_bound: Val - RMSE`.
4. Generate Narrative: Use `nx.has_path` and `nx.shortest_path(G, source=shock_node, target=target)` to locate the driving timeline array. Build an english string tracing the path.
5. Return JSON payload string using `json.dumps()`.
</action>
<read_first>
`src/simulation/engine.py`
</read_first>
<acceptance_criteria>
`cat src/xai/explainer.py | grep "generate_xai_payload"` returns a match.
`cat src/xai/explainer.py | grep "RMSE"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create `scripts/test_xai.py`. Integrate Stages 1 -> 4, and pass the output `sim_out`, `G`, and `models_dict` into the XAI explainer. Print the strictly nested JSON payload natively.
</action>
<read_first>
`scripts/test_sim.py`
</read_first>
<acceptance_criteria>
Running `python scripts/test_xai.py` executes successfully returning JSON.
</acceptance_criteria>
</task>

## Verification
- JSON Output payload includes calculated static confidence intervals (bounds arrays explicitly displayed).
- The text generator correctly traces back through DAG layers using node variables correctly.

## Must Haves
- Networkx `shortest_path` traversal ensuring node logical linkage.
- Error bounds successfully populated using baseline Phase 2 constraints.
