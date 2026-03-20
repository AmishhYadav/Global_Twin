# Requirements — v2.0

## Data Ingestion (DATA)
- [ ] **DATA-03**: System fetches real-world economic indicators from FRED and Yahoo Finance APIs.
- [ ] **DATA-04**: Raw data is normalized and stored as clean CSVs in `data/raw/`.
- [ ] **DATA-05**: System supports 5 economies (US, EU, China, India, Japan) with per-country indicator sets.
- [ ] **DATA-06**: Unified data loader queries by country and indicator name.

## Machine Learning (ML)
- [ ] **ML-02**: Feature engineering includes cross-country correlations and global composite indicators.
- [ ] **ML-03**: System trains multiple model types (RF + Gradient Boosting) per target variable.
- [ ] **ML-04**: Best model auto-selected per target; model registry stores metadata and metrics.

## Knowledge Graph & Simulation (SIM)
- [ ] **SIM-06**: Knowledge graph spans multiple countries with cross-border causal edges (50+ nodes).
- [ ] **SIM-07**: Pre-built scenarios (Oil Embargo, Trade War, Pandemic, Rate Hike) apply multi-variable shocks.
- [ ] **SIM-08**: Custom scenario builder allows user-defined multi-variable shock combinations.

## Dashboard & Visualization (UI)
- [ ] **UI-04**: Multi-tab dashboard with World View, Country Deep-Dive, Scenario Lab, and Graph Explorer.
- [ ] **UI-05**: World View shows global heatmap; Country view filters by economy.

## Infrastructure (INFRA)
- [ ] **INFRA-01**: Graceful error handling for API failures, loading states, and deployment readiness.

## Traceability
- **Phase 8**: DATA-03, DATA-04
- **Phase 9**: DATA-05, DATA-06
- **Phase 10**: ML-02
- **Phase 11**: ML-03, ML-04
- **Phase 12**: SIM-06
- **Phase 13**: SIM-07, SIM-08
- **Phase 14**: UI-04, UI-05
- **Phase 15**: INFRA-01
