# Features Research

**Domain:** Global Digital Twin AI – Supply Chain & Energy Impact Simulator
**Analysis Type:** Greenfield Feature Mapping

## 1. Table Stakes (Must Haves)
Without these, users will abandon the product.
- **Data Ingestion UI**: Simple file upload interface for CSV/Excel data with column mapping (e.g., mapping user columns to "Date", "Oil Price", "Inflation").
- **Dependency Visualization**: A visual graph showing the nodal relationships (e.g., Oil Price node connected to Shipping node connected to Retail Price node).
- **Time-Series Baseline**: Visualizing historical data trends alongside baseline future predictions without any shocks.
- **Scenario Simulator ("What-if")**: Input form to apply a "shock" (e.g., +20% spike in Oil Price at Month 3).
- **Impact Comparison**: Side-by-side charts of "Baseline vs. Shock Output".

## 2. Differentiators (Competitive Advantage)
These features drive adoption and wow factor.
- **Explainable AI (XAI)**: When the model predicts shipping costs will rise, providing a text explanation of *why* (e.g., "Driven by historical correlation: a 10% oil spike historically shifts shipping by 4% within 2 weeks").
- **Confidence Scores**: Displaying prediction intervals (e.g., "Predicted inflation: +1.2% ±0.3% with 90% confidence").
- **Cascading Pathway Highlight**: In the network graph, visually highlighting the "path of impact" as a shock propagates from node to node over time.

## 3. Anti-Features (What NOT to build)
- **Live API Integration (for MVP)**: Managing API rate limits, schema changes, and backfills will stall the core value delivery. Stick to static data.
- **Fully Automated "Hands-off" AI**: The "black box" approach. Users need to trace the logic step-by-step; they won't trust an unexplained number.
- **Micro-level factory tracking**: Do not try to model individual trucks or factories. Keep modeling at macroeconomic/industry variable levels.
