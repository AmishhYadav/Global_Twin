# Global Digital Twin AI – Supply Chain & Energy Impact Simulator

## What This Is

An AI-powered simulation system that ingests economic and supply chain data (like oil prices, shipping costs, inflation) to model their interconnected relationships. It allows users to run "what-if" scenarios to predict the cascading ripple effects of global events on product prices and inflation, visualized through an interactive dashboard.

## Core Value

Accurately model interconnected global systems to predict the ripple effects of economic and supply chain changes across industries.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] Ingest real-world historical data via static CSV/Excel uploads. *(Validated in Phase 1: Data Ingestion Foundation)*
- [x] Train machine learning models (Random Forest + time-series) to learn variable relationships. *(Validated in Phase 2: Core ML Modeling)*
- [x] Build a lightweight in-memory knowledge graph (e.g., NetworkX) to represent dependencies (oil → shipping → products). *(Validated in Phase 3: Knowledge Graph Structure)*
- [x] Predict cascading effects and ripple impacts when a root variable changes. *(Validated in Phase 4: Simulation Engine Execution)*

### Active

<!-- Current scope. Building toward these. -->

- [ ] Provide an interactive dashboard with graphs and visualizations for "what-if" simulations.
- [ ] Display real-time predictions with AI explanations and confidence scores.

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- [Live API Data Integration] — Deferred to v2. MVP will focus on static CSVs for faster iteration and proving the ML model.
- [Dedicated Graph Database (Neo4j)] — Deferred to v2. A lightweight in-memory Python graph is sufficient for MVP complexity and keeps the stack simple.

## Context

- **Technical Environment:** Python-heavy backend for ML and graph modeling. Frontend will need a rich interactive dashboard (likely React/Next.js or Streamlit/Dash if staying purely in Python).
- **Core ML Need:** Needs to handle time-series forecasting combined with graph-based dependency cascading.

## Constraints

- **Tech Stack**: Keep it simple for MVP. Use NetworkX instead of a heavy Graph DB. Use CSVs instead of complex API pipelines.
- **Timeline**: Fast time-to-value for the MVP by strictly limiting data ingestion scope.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use CSVs for MVP Data | Fastest path to prove the ML relationships and UI without getting bogged down in API rate limits and pipelines. | — Pending |
| NetworkX for Knowledge Graph | Avoids operational overhead of deploying and managing a Graph Database like Neo4j for the initial proof-of-concept. | — Pending |

---
*Last updated: 2026-03-19 after initialization*
