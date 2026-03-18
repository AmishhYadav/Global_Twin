# Architecture Research

**Domain:** Global Digital Twin AI – Supply Chain & Energy Impact Simulator
**Analysis Type:** Greenfield System Design

## 1. System Pattern
For an MVP focused on data science and simulation, a **Modular Monolith** pattern is recommended. The backend handles data ingestion, model training, graph traversal, and API serving in a single Python application to simplify development.

## 2. Component Boundaries
- **Data Ingestion Layer**: Parses CSVs, normalizes date formats, handles missing values, and aligns time-series frequencies.
- **Modeling Engine**: Wraps Scikit-learn (Random Forest) to train relationships between columns.
- **Graph Engine**: Uses NetworkX to maintain the directed graph of dependencies (edges signify correlation/causality learned by the Modeling Engine).
- **Simulation Engine**: Takes a "shock" input, traverses the graph via the Modeling Engine, and computes delta changes over time.
- **API Server**: FastAPI layer exposing REST endpoints (`/upload`, `/train`, `/simulate`).
- **Frontend SPA**: React application that consumes the API and renders charts/graphs.

## 3. Data Flow
1. **Upload**: User sends CSV -> API Server -> Data Ingestion Layer formats and saves locally.
2. **Train**: User triggers training -> Modeling Engine builds Random Forest regressor -> Graph Engine instantiates structure -> Insights returned.
3. **Simulate**: User sends shock parameters -> Simulation Engine runs step-wise cascading predictions -> Returns Baseline vs. Shock timeseries.

## 4. Suggested Build Order
1. Data Ingestion & Normalization scripts.
2. Modeling & Graph Engine (the core logic).
3. Simulation Engine script (validating "what-if" logic).
4. FastAPI wrapper around the engines.
5. Frontend Dashboard to consume endpoints.
