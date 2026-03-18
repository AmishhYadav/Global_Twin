# 🌍 Global Twin: Supply Chain & Economic Simulator

**Global Twin** is an intelligent, transparent Simulation Engine designed to mathematically map macroeconomic indices, supply-chain costs, and dynamic dependencies. It combines powerful **Random Forest** machine learning with **Knowledge Graph** architectures to allow users to visually trace and execute cascading "what-if" scenarios across the global market.

## 🚀 Key Features

* **Causal Knowledge Graph:** Extracts explicit causal dependencies between global variables automatically, rendering them natively into a physics-based interactive topology using `Pyvis`.
* **Explainable AI (XAI) Engine:** Every dynamic "Shock" (e.g., *Oil Prices +20%*) cascades across the structural DAG. The system outputs mathematically grounded ±RMSE boundaries and pure-English XAI traceback narrations explaining *why* the metrics shifted sequentially.
* **Interactive Dashboard:** Built natively on `Streamlit` and `Plotly`, users can adjust origin shocks via sliders and watch multi-step Trajectory deviations (`T+3` horizon) render instantly. 

## 🛠️ Architecture

1. **Phase 1-2 (ML Backend):** Time-series ingestion, feature engineering (rolling windows, lags), and Random Forest multi-target regression.
2. **Phase 3-4 (Topology Generation):** Structural Directed Acyclic Graph generation dynamically bounding the deepest predictive coefficients above `5%` importance.
3. **Phase 5 (XAI Mapping):** Tracing `nx.shortest_path()` network routes and establishing statistical ±RMSE thresholds shaping explainable textural payload dictionaries.
4. **Phase 6-7 (Dashboard GUI):** Native `Streamlit` plotting intercepting outputs into dynamic Interactive Physics graphs (`components.v1.html`) and bound UI Plotly charts mapping `baseline` traces against `shocked` traces effectively.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/global-twin.git
   cd global-twin
   ```
2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Dashboard Simulator:**
   ```bash
   streamlit run src/dashboard/app.py
   ```

## 🗺️ Roadmap & Trajectory

This project traces its execution methodology strictly via an Agentic `GSD (Get Shit Done)` framework residing within `.planning/`. See `.planning/PROJECT.md` for our historical constraints and architectural milestones cleanly defining the `v1.0` iteration capabilities!
