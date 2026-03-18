# Project Roadmap

**7 phases** | **10 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Criteria |
|---|-------|------|--------------|----------|
| 1 | Data Ingestion Foundation | Establish parsing and normalization of historical datasets | DATA-01, DATA-02 | 3 |
| 2 | Core ML Modeling | Train Random Forest models to learn dependencies between variables | ML-01 | 4 |
| 3 | Knowledge Graph Structure | Build NetworkX in-memory graph to represent dependencies | SIM-01 | 3 |
| 4 | Simulation Engine Execution | Cascade "what-if" shocks through the graph | SIM-02, SIM-03 | 2 |
| 5 | Output & XAI Explainability | Provide confidence bounds and text-based path explanations | SIM-04, SIM-05 | 2 |
| 6 | Dashboard Layout & Charting | Build React/Next.js UI to compare baseline vs shock outputs | UI-01 | 3 |
| 7 | Interactive Graph UI | Render the NetworkX graph in the UI and show ripple propagation | UI-02, UI-03 | 3 |

---

## Phase Details

### Phase 1: Data Ingestion Foundation
**Goal**: Establish parsing and normalization of historical CSV datasets
**Requirements**: DATA-01, DATA-02
**Success Criteria**:
1. User can successfully upload a CSV via REST endpoint or local script.
2. System normalizes date formats and handles missing values silently or with warnings.
3. System outputs a clean, unified Pandas DataFrame ready for ML training.

### Phase 2: Core ML Modeling
**Goal**: Train Random Forest models to learn dependencies between variables
**Requirements**: ML-01
**Success Criteria**:
1. System splits data temporally without leakage.
2. Model successfully predicts T+1 with baseline evaluation metrics logged (RMSE, MAE, R²).
3. Feature importance analysis is performable for each relationship to establish graph nodes.
4. Basic explainability reports can be generated for the model's predictions.

### Phase 3: Knowledge Graph Structure
**Goal**: Build NetworkX in-memory graph to represent dependencies
**Requirements**: SIM-01
**Success Criteria**:
1. Directed graph is built automatically from the ML feature importances.
2. Graph nodes are variables; edges have weights representing correlation strength.
3. Graph can be queried for child/parent node relationships.

### Phase 4: Simulation Engine Execution
**Goal**: Cascade "what-if" shocks through the graph
**Requirements**: SIM-02, SIM-03
**Success Criteria**:
1. System accepts a root node shock (e.g., Variable X = +20%).
2. Engine computes multi-step cascading impacts downstream over time (T+1, T+2...).

### Phase 5: Output & XAI Explainability
**Goal**: Provide confidence bounds and text-based path explanations
**Requirements**: SIM-04, SIM-05
**Success Criteria**:
1. Output payload includes confidence intervals.
2. Generator creates a plain-text explanation of the strongest propagation path.

### Phase 6: Dashboard Layout & Charting
**Goal**: Build React UI to compare baseline vs shock outputs
**Requirements**: UI-01
**Success Criteria**:
1. Frontend pulls simulation output from the backend API.
2. Line charts render Baseline alongside Shock trajectory.
3. User can input shock parameters from a web form.

### Phase 7: Interactive Graph UI
**Goal**: Render the NetworkX graph in the UI and show ripple propagation
**Requirements**: UI-02, UI-03
**Success Criteria**:
1. Graph is rendered cleanly using Vis.js or Plotly.
2. Nodes and edges adjust visually to represent the shock's path.
3. System responds smoothly to graph drill-downs.
