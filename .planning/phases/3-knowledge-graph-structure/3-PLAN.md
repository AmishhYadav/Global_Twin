---
wave: 1
depends_on: []
files_modified: ["requirements.txt", "src/graph/build_graph.py", "scripts/test_graph.py"]
autonomous: true
---

# Plan 1: Knowledge Graph Topology

## Objective
Automatically construct an exported NetworkX DAG (JSON) using the feature importances extracted from the Phase 2 ML models.

## Tasks

<task>
<action>
Update `requirements.txt` to append `networkx==3.2.1`.
</action>
<read_first>
`requirements.txt`
</read_first>
<acceptance_criteria>
`cat requirements.txt | grep networkx` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `src/graph/` and build `src/graph/build_graph.py`. 
Implement `create_knowledge_graph(ml_results_dict, importance_threshold=0.05)`.
Logic:
1. Initialize `nx.DiGraph()`.
2. Iterate `ml_results` targets and top features. Add edges from Feature Node -> Target Node with `weight` = importance. Only add edges if `importance > importance_threshold`.
3. Check DAG compliance: while `not nx.is_directed_acyclic_graph(G)`, extract `nx.simple_cycles(G)`. For each cycle, identify the edge with the lowest weight and remove it `G.remove_edge(u, v)`.
4. Export the final DAG into a dictionary using `nx.node_link_data(G)` and save to a JSON file at `models/graph.json`.
</action>
<read_first>
`src/models/train.py`
</read_first>
<acceptance_criteria>
`cat src/graph/build_graph.py | grep "\.is_directed_acyclic_graph"` returns a match.
`cat src/graph/build_graph.py | grep "node_link_data"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create `scripts/test_graph.py`. Import and run the pipelines from `test_ml` generating the `results` dictionary. Pass it to `create_knowledge_graph(results)`. Then load the output `models/graph.json` back into memory and assert `len(json['nodes']) > 0`.
</action>
<read_first>
`scripts/test_ml.py`
</read_first>
<acceptance_criteria>
Running `python scripts/test_graph.py` creates `models/graph.json` and exits with code 0.
</acceptance_criteria>
</task>

## Verification
- Directed graph is built automatically from the ML feature importances (Threshold enforced).
- System strictly builds a DAG, deleting edges if a cyclic loop forms.
- Graph can be loaded from the JSON output payload perfectly without memory dependencies.

## Must Haves
- Python script outputs a standardized JSON payload.
- Strict DAG compliance guarantee.
- NetworkX underlying mechanics.
