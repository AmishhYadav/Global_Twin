# Phase 3: Knowledge Graph Structure - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase translates the machine learning feature importances (generated in Phase 2) into an in-memory knowledge graph. It establishes the causal connections between targeted economic and supply chain variables, enabling downstream cascading impact simulations.
(Simulating the cascade belongs in Phase 4.)
</domain>

<decisions>
## Implementation Decisions

### Graph Topology
- Construct a strictly Directed Acyclic Graph (DAG) to guarantee stable and finite cascading simulations. Break any detected cycles by removing the weakest importance edge.

### Edge Creation Rule
- Set a hard threshold: only features with an importance score > 5% (0.05) are transformed into directed edges (Feature -> Target).

### Graph Storage
- Produce an automated builder script that constructs the graph artifact post-ML training. Save this artifact as a JSON structure so it can be loaded instantly during simulations without re-accessing ML state.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Core MVP constraints preventing Neo4j stack drift.
- `.planning/REQUIREMENTS.md` — Simulation topology requirement (SIM-01).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/models/train.py` — Outputs feature importance arrays which serve as the input for this graph builder.

### Established Patterns
- Python standard library + NetworkX for topology management.
</code_context>

<specifics>
## Specific Ideas
- Implement a cycle-breaking check using NetworkX (`nx.simple_cycles` or `nx.is_directed_acyclic_graph`) to ensure strict DAG before JSON export.
</specifics>

<deferred>
## Deferred Ideas
None.
</deferred>

---

*Phase: 3-knowledge-graph-structure*
*Context gathered: 2026-03-19*
