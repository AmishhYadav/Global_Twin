---
wave: 1
depends_on: []
files_modified: ["src/simulation/engine.py", "scripts/test_sim.py", "src/models/train.py"]
autonomous: true
---

# Plan 1: Knowledge Graph Simulator

## Objective
Implement a multi-step temporal cascade Engine passing historical state datasets organically through the existing Random Forest array across a user-defined `T+3` window. 

## Tasks

<task>
<action>
Update `src/models/train.py` temporarily to save the `feature_cols` list explicitly inside the returned model dictionary (e.g. `results[target]['feature_names'] = feature_cols`), ensuring the simulator passes the exact ordered feature array layout to `.predict()`.
</action>
<read_first>
`src/models/train.py`
</read_first>
<acceptance_criteria>
`cat src/models/train.py | grep "feature_names"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `src/simulation/` and build `src/simulation/engine.py`. 
Implement `run_simulation(models_dict, historical_df, shock_node, shock_pct, horizon=3)`.
Logic:
1. Copy `historical_df` into `base_df` and `shock_df`.
2. Grab the literal last index of `shock_df` and apply the shock multiplier: `shock_df.loc[last_idx, shock_node] *= (1 + shock_pct)`.
3. Loop 1 to `horizon`:
   - Compute features using `create_time_series_features()` on both dfs. Extract the single terminal row `[-1:]` from both.
   - For every `target` in `models_dict`: Make identical `rf.predict()` calls extracting next day's values.
   - Synthesize a new dummy row mapping the new target variables identically to both dataframes. Append using `pd.concat` and let loop index slide forward mathematically!
4. Return differential trajectories.
</action>
<read_first>
`src/features/build_features.py`
</read_first>
<acceptance_criteria>
`cat src/simulation/engine.py | grep "def run_simulation"` returns a match.
`cat src/simulation/engine.py | grep "shock_pct"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create `scripts/test_sim.py`. Link Stages 1 (Ingest) and 2 (Train). Extract the final historical mock data, shock the parameter `Oil_Price` by +20% (`0.20`), and call `run_simulation`. Print out the comparative values across T+1, T+2, T+3.
</action>
<read_first>
`scripts/test_ml.py`
</read_first>
<acceptance_criteria>
Running `python scripts/test_sim.py` executes successfully.
</acceptance_criteria>
</task>

## Verification
- Recursive computation executes flawlessly using Random Forest states across standard intervals.
- The output clearly highlights deviation trajectory when shocked.

## Must Haves
- Python integration with actual feature creation rolling averages.
- Hard output delta array tracking predictions.
