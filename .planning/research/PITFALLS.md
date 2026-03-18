# Pitfalls Research

**Domain:** Global Digital Twin AI – Supply Chain & Energy Impact Simulator
**Analysis Type:** Greenfield Gotchas & Risks

## 1. The "Black Box" Rejection
- **Warning Sign**: The model outputs a prediction, but the user cannot understand why.
- **Prevention**: Incorporate feature importance (SHAP values) from the Random Forest model into the UI. Always show a confidence band alongside the point prediction.
- **Phase Mapping**: Core modeling phase and UI Dashboard phase.

## 2. Temporal Leakage in Time-Series
- **Warning Sign**: Model performance looks flawless during validation but fails completely when simulating forward.
- **Prevention**: Use strict temporal split for train/test data. Do not use random K-fold on time-series. Ensure lagged variables are strictly enforced (e.g., predicting T+1 using only data up to T).
- **Phase Mapping**: Data Ingestion and Modeling phase.

## 3. Infinite Feedback Loops
- **Warning Sign**: When doing a multi-step graph traversal, variables bounce back and forth (A increases B, which increases A) causing exponential blowouts.
- **Prevention**: In the Simulation Engine, implement dampening/decay factors on multi-step ripples, and limit traversal depth (e.g., max 3 hops) to mimic real-world friction.
- **Phase Mapping**: Simulation Engine implementation.

## 4. UI Overwhelm
- **Warning Sign**: A "spaghetti graph" with 100+ nodes and lines that looks impressive but is unusable.
- **Prevention**: Only visualize the nodes that pass a significance threshold for the current simulation. Implement interactive drill-downs.
- **Phase Mapping**: Frontend Dashboard phase.
