# Phase 6: Dashboard Layout & Charting - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase wraps the backend Simulation Sequence and XAI narrative payloads inside a graphical user interface (GUI). It exposes the mathematically validated causal graph controls directly to the end-user via interactive sliding scales.
</domain>

<decisions>
## Implementation Decisions

### Framework Choice
- Adopt `Streamlit` as the core GUI framework natively bridging raw Python ML dictionaries into reactive DOM elements automatically.

### Layout Structure
- Utilize a classic Left-Sidebar for mathematical input definition (Target Variable dropdown, Shock percentage slider).
- Render the main central container with output visualizations, enabling immediate user feedback.

### Chart Library
- Integrate `Plotly Express` and `Plotly.graph_objects` to securely plot dynamic line charts charting original baseline arrays vs shocked trajectory deviations dynamically with interactive hover-tooltips.
</decisions>

<canonical_refs>
## Canonical References
- `.planning/PROJECT.md` — Constraint: UI-01, UI-02, UI-03.
</canonical_refs>

<code_context>
## Existing Code Insights
- `src/xai/explainer.py` — Outputs `json` string payload containing all data required for rendering visually.
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

*Phase: 6-dashboard-layout-charting*
*Context gathered: 2026-03-19*
