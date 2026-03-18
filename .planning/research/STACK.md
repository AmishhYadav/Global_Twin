# Stack Research

**Domain:** Global Digital Twin AI – Supply Chain & Energy Impact Simulator
**Analysis Type:** Greenfield Standard Stack (2025)

## 1. Programming Languages
- **Python (3.12+)**: Absolute table stakes for ML, time-series, and data engineering. Essential for pandas, scikit-learn, and network modeling.
- **TypeScript**: Standard for the interactive frontend dashboard to ensure type safety in complex data visualizations.

## 2. Core Frameworks & Libraries
### Data Science & Modeling
- **Pandas / Polars**: Polars is recommended for large datasets and fast CSV processing; Pandas is the fallback for broader ecosystem compatibility.
- **Scikit-learn**: For Random Forest models to learn variable dependencies.
- **NetworkX**: Standard Python graph library. Perfect for traversing dependencies without needing a full graph database.
- **Prophet / Darts**: For baseline time-series forecasting.

### Backend App & API
- **FastAPI**: The standard for serving Python ML models. Excellent async support and automatic OpenAPI docs.
- **Celery / RQ**: Task queues for running long "what-if" simulations in the background without blocking the UI.

### Frontend Dashboard
- **React + Next.js**: Standard for deep, interactive web applications.
- **Recharts / Plotly.js / Vis.js**: Plotly or Vis.js are essential for rendering interactive network graphs and time-series impact charts.

## 3. Storage & Infrastructure
- **MVP Data Storage**: Local file system (CSV/Excel) as requested.
- **Cache**: Redis is highly recommended for caching simulation results and powering the task queue.

## 4. What NOT to Use
- **Neo4j / Gremlin**: Explicitly out of scope per user request, but also massive overkill for an MVP.
- **Deep Learning / PyTorch**: For initial time-series and relationship mapping, starting with Random Forest is more explainable and faster to build.

---
*Confidence Level: High. This stack optimizes for rapid ML prototyping while building a foundation for production.*
