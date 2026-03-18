# Requirements

## v1 Requirements

### Data Ingestion (DATA)
- [ ] **DATA-01**: User can upload static CSV/Excel files containing historical time-series data.
- [ ] **DATA-02**: System automatically parses, normalizes dates, and handles basic missing values.

### Machine Learning (ML)
- [ ] **ML-01**: System trains a Random Forest + Time-Series model to learn variable dependencies based on historical data.

### Knowledge Graph & Simulation (SIM)
- [ ] **SIM-01**: System builds a lightweight in-memory directed graph (NetworkX) representing dependencies (e.g., Oil -> Shipping).
- [ ] **SIM-02**: User can input a "what-if" scenario / shock (e.g., Oil +20% spike at Month 3).
- [ ] **SIM-03**: System cascades the shock through the graph to predict downstream impact over time.
- [ ] **SIM-04**: Predictions include a confidence score or interval.
- [ ] **SIM-05**: System provides a text-based XAI explanation of the propagation path.

### Dashboard & Visualization (UI)
- [ ] **UI-01**: Display baseline future predictions vs shock output in interactive charts.
- [ ] **UI-02**: Display an interactive network graph of the variables.
- [ ] **UI-03**: Highlight the cascading pathway on the graph when a simulation runs.

## v2 Requirements (Deferred)
- Live API integration for real-time automated data ingest (AlphaVantage, World Bank, etc.).
- Dedicated scalable Graph Database (Neo4j) for massive industry-scale networks.

## Out of Scope
- **Micro-level factory/truck tracking:** Model operates at a macroeconomic/industry-variable level, not at a discrete simulation level.

## Traceability
*(To be updated by roadmap)*
