# Research Summary

**Domain:** Global Digital Twin AI – Supply Chain & Energy Impact Simulator

## Key Findings

**Stack:**
- **Backend:** Python 3.12+, FastAPI, Celery
- **Modeling & Data:** Pandas, Scikit-learn (Random Forest), NetworkX
- **Frontend:** React + Next.js, Vis.js or Plotly for graph/chart visualization
- **Storage:** Local CSVs (MVP), Redis for tasks

**Table Stakes (Must-Haves):**
- Data ingestion UI for CSV uploads
- Visual node graph of relationships
- Time-series baseline + "what-if" shock scenario simulator
- Side-by-side outcome charts

**Differentiators:**
- Explainable AI text ("Why did this go up?")
- Confidence bands on predictions
- Highlighting the propagation path on the node graph

**Watch Out For:**
- **Explainability:** Users reject "black-box" numbers. Prove paths.
- **Time Leakage:** Strict temporal splitting for training data.
- **Feedback Loops:** Dampen multi-step ripple propagation to avoid infinite scaling blowouts.
- **Visual Overwhelm:** Don't render a 100-node spaghetti graph; allow drill-downs.
