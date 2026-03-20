# Global Twin — Project Definition

## Vision
A real-world macroeconomic simulation engine that models interconnected global systems across 5 major economies (US, EU, China, India, Japan) to predict the cascading ripple effects of economic shocks.

## Current State
- **v1.0** shipped (2026-03-19): End-to-end proof-of-concept with synthetic data, 3 variables, RF models, DAG graph, Streamlit dashboard.
- **v2.0** in progress: Real-world data, 5 economies, 15+ indicators, multi-model pipeline, scenario engine, redesigned dashboard.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] Ingest real-world historical data via static CSV/Excel uploads. *(v1.0)*
- [x] Train ML models to learn variable relationships. *(v1.0)*
- [x] Build in-memory knowledge graph representing dependencies. *(v1.0)*
- [x] Predict cascading effects when a root variable changes. *(v1.0)*
- [x] Display predictions with AI explanations and confidence scores. *(v1.0)*
- [x] Interactive dashboard with charts for "what-if" simulations. *(v1.0)*
- [x] Interactive network diagram of the knowledge graph. *(v1.0)*

### Active

<!-- Current scope — v2.0. -->

- [ ] Fetch real economic indicators from FRED & Yahoo Finance APIs.
- [ ] Support 5 economies with per-country indicator sets.
- [ ] Cross-country feature correlations and global composites.
- [ ] Multi-model ML pipeline (RF + Gradient Boosting) with auto-selection.
- [ ] Cross-country knowledge graph with 50+ nodes.
- [ ] Pre-built macro shock scenarios (Oil Embargo, Trade War, Pandemic, Rate Hike).
- [ ] Multi-tab dashboard: World View, Country Dive, Scenario Lab, Graph Explorer.

### Out of Scope

- Micro-level factory/truck tracking (model operates at macroeconomic level).
- Real-time streaming data (batch fetch is sufficient for v2.0).
- Dedicated graph database (NetworkX in-memory is sufficient at current scale).
