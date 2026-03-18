# Phase 4: Simulation Engine Execution - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase implements the temporal prediction cascade. The simulation engine iterates future step intervals (`T+1`, `T+2`..) recursively appending calculated data back into the dataset array, causing rolling features (built in Phase 2) to organically drag the shock multiplier forward into all dependent ML Random Forest predictions dynamically.
</domain>

<decisions>
## Implementation Decisions

### Shock Definition
- Implement percentage-based shocks (`Variable = Variable * (1 + 0.20)` for +20%) directly onto the terminal end (`T=0`) of the historical sliding window DataFrame.

### Temporal Cascade Horizon
- Confine the temporal cascade generator safely to `T+3`. This limits cumulative prediction drift (where ML estimates build blindly on other ML estimates indefinitely).

### Simulation Computation
- Utilize the actual Phase 2 `RandomForest` machine learning agents natively. Because the ML architecture evaluates feature spaces independently across the whole dataframe block, there's no need to explicitly traverse the DAG edge-by-edge. Instead, we can just step the entire DataFrame 1 day forward iteratively 3 times, predicting all ML targets on each cycle, regenerating the moving averages between loops.
</decisions>

<canonical_refs>
## Canonical References

### Project Scope
- `.planning/PROJECT.md` — Requirement (SIM-02 & SIM-03).
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/features/build_features.py` — Critical execution dependency for the simulator. The simulator extracts baseline and custom shocked tracking DataFrames, and must call this feature generation tool on every loop (Dropping old rows dynamically) to ensure all `.shift()` and `.rolling()` indicators ingest the mock simulation data optimally.
</code_context>

<specifics>
## Specific Ideas
- Generate an output JSON dictionary returning two parallel arrays: `Baseline_Forecast`, and `Shocked_Forecast`.
</specifics>

<deferred>
## Deferred Ideas
None.
</deferred>

---

*Phase: 4-simulation-engine-execution*
*Context gathered: 2026-03-19*
