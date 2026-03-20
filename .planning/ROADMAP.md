# Project Roadmap

**v1.0** — [Archived](milestones/v1.0-ROADMAP.md) ✓ (7 phases, completed 2026-03-19)

---

## v2.0 — Global Twin Expansion

**8 phases** | **12 requirements mapped** | Target: Real-world multi-country simulation

| # | Phase | Goal | Requirements | Criteria |
|---|-------|------|--------------|----------|
| 8 | Real-World Data Integration | Pull 15+ real economic indicators from FRED & Yahoo Finance | DATA-03, DATA-04 | 3 |
| 9 | Multi-Country Data Architecture | Restructure data layer for 5 economies (US, EU, CN, IN, JP) | DATA-05, DATA-06 | 3 |
| 10 | Enhanced Feature Engineering | Cross-country correlations, inter-sector features, global composites | ML-02 | 3 |
| 11 | Multi-Model ML Pipeline | RF + Gradient Boosting with model comparison and registry | ML-03, ML-04 | 4 |
| 12 | Cross-Country Knowledge Graph | Massive DAG with intra-country AND cross-border causal edges | SIM-06 | 3 |
| 13 | Scenario Engine | Pre-built macro shock scenarios (Oil Embargo, Trade War, Pandemic, Rate Hike) | SIM-07, SIM-08 | 3 |
| 14 | Dashboard Redesign | Multi-tab layout: World View, Country Dive, Scenario Lab, Graph Explorer | UI-04, UI-05 | 4 |
| 15 | Polish, Docs & Deploy | Error handling, loading states, README, deployment config | INFRA-01 | 2 |

---

## Phase Details

### Phase 8: Real-World Data Integration
**Goal**: Pull ~15 real economic indicators from FRED and Yahoo Finance APIs
**Requirements**: DATA-03, DATA-04
**Success Criteria**:
1. Indicator registry defines all tracked variables with source API and ticker/series ID.
2. Fetcher downloads 5+ years of daily/monthly data per indicator.
3. Raw data saved as normalized CSVs in `data/raw/`.

### Phase 9: Multi-Country Data Architecture
**Goal**: Restructure data layer for 5 economies (US, EU, China, India, Japan)
**Requirements**: DATA-05, DATA-06
**Success Criteria**:
1. Each country has its own indicator set with standardized column naming.
2. Cross-country reference table links equivalent indicators.
3. Unified loader can query by country + indicator.

### Phase 10: Enhanced Feature Engineering
**Goal**: Cross-country correlations, inter-sector dependencies, global composites
**Requirements**: ML-02
**Success Criteria**:
1. Features include cross-country lag correlations (e.g., US rate → INR/USD).
2. Global composite indicators computed (e.g., weighted GDP index).
3. Feature matrix supports 50+ columns per country.

### Phase 11: Multi-Model ML Pipeline
**Goal**: RF + Gradient Boosting with model comparison and registry
**Requirements**: ML-03, ML-04
**Success Criteria**:
1. Both RF and GBR trained per target variable.
2. Best model auto-selected by validation RMSE.
3. Model registry stores metadata, metrics, and feature names.
4. Comparison report generated per target.

### Phase 12: Cross-Country Knowledge Graph
**Goal**: Massive DAG spanning all countries with cross-border causal edges
**Requirements**: SIM-06
**Success Criteria**:
1. Graph has 50+ nodes (variables × countries).
2. Edges include both intra-country and cross-border dependencies.
3. DAG constraint enforced with cycle breaking.

### Phase 13: Scenario Engine
**Goal**: Pre-built macro shock scenarios
**Requirements**: SIM-07, SIM-08
**Success Criteria**:
1. Scenario registry defines 4+ named scenarios with multi-variable shocks.
2. User selects a scenario and system applies all shocks simultaneously.
3. Custom scenario builder allows mixing variables and magnitudes.

### Phase 14: Dashboard Redesign
**Goal**: Multi-tab Streamlit layout with world view, country drill-down, scenario lab
**Requirements**: UI-04, UI-05
**Success Criteria**:
1. World View tab shows global heatmap of variable states.
2. Country Deep-Dive tab filters charts by selected economy.
3. Scenario Lab tab offers pre-built + custom shock builder.
4. Graph Explorer tab renders full cross-country DAG.

### Phase 15: Polish, Docs & Deploy
**Goal**: Production readiness with error handling and documentation
**Requirements**: INFRA-01
**Success Criteria**:
1. Graceful error handling for API failures and missing data.
2. Loading states and progress bars in dashboard.
