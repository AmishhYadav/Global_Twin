# 🌍 Global Twin v2.0

**Multi-country macroeconomic simulation engine with AI-powered causal analysis.**

Simulate the cascading ripple effects of economic shocks across 5 major economies — powered by machine learning, knowledge graphs, and pre-built macro scenarios.

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

### 🌍 Multi-Country Modeling
Model 5 major economies simultaneously: **US**, **EU**, **China**, **India**, **Japan** — with 30+ real-world economic indicators including GDP, unemployment, inflation, interest rates, commodities, and exchange rates.

### 🧠 Multi-Model ML Pipeline
- **Random Forest** and **Gradient Boosting** trained per target variable
- Auto-selection by lowest validation RMSE
- 200+ engineered features (cross-country lags, global composites, inter-sector spreads)
- Model registry with full metadata and comparison reports

### 🕸️ Cross-Country Knowledge Graph
- Directed Acyclic Graph (DAG) with 50+ nodes
- ML-derived edges from feature importances
- 30+ structural economic relationships (e.g., `US_FED_RATE → INR_USD`)
- Nodes tagged by country and sector for filtering

### 🔬 Pre-Built Macro Scenarios
| Scenario | Severity | Key Shocks |
|----------|----------|------------|
| 🛢️ Oil Embargo | High | Oil +60%, Gas +40%, Shipping +30% |
| ⚔️ US-China Trade War | High | CN GDP -5%, CNY +8%, Semiconductors -10% |
| 🦠 Global Pandemic | Critical | GDP collapse, VIX +150%, Oil -40% |
| 📈 Fed Rate Hike | Medium | Fed Rate +40%, EM currencies depreciate |
| 🌱 Energy Transition | Medium | Oil -30%, Copper +40% |
| 🇨🇳 China Slowdown | High | CN GDP -10%, Copper -25% |

### 📊 Interactive Dashboard
4-tab Streamlit dashboard with premium dark theme:
- **World View** — Global metrics, commodity trends, GDP comparison
- **Country Deep-Dive** — Per-economy selector with historical charts
- **Scenario Lab** — Pre-built + custom shock simulation with XAI
- **Graph Explorer** — Interactive Pyvis DAG color-coded by country

---

## 🏗️ Architecture

```
src/
├── data/
│   ├── indicators.py        # Registry of 30 real-world indicators (FRED + Yahoo)
│   ├── fetch.py             # API data fetchers (FRED CSV + yfinance)
│   ├── country_manager.py   # Multi-country data layer with cross-country queries
│   └── ingest.py            # v1.0 CSV ingestion (legacy)
├── features/
│   └── build_features.py    # 4-layer feature engineering pipeline
├── models/
│   ├── train.py             # RF + Gradient Boosting with auto-selection
│   └── registry.py          # Model persistence (pickle + JSON metadata)
├── graph/
│   └── build_graph.py       # Cross-country DAG with structural edges
├── simulation/
│   ├── engine.py            # Multi-variable shock cascade engine
│   └── scenarios.py         # 6 pre-built scenarios + custom builder
├── xai/
│   └── explainer.py         # Confidence bounds + causal narratives
└── dashboard/
    └── app.py               # 4-tab Streamlit dashboard
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/Global_Twin.git
cd Global_Twin
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Dashboard (Synthetic Data)
```bash
streamlit run src/dashboard/app.py
```
No API keys needed — the dashboard launches immediately with synthetic data that simulates realistic economic behavior.

### 3. (Optional) Fetch Real Data
```bash
python scripts/fetch_data.py
```
This downloads 5+ years of real data from FRED and Yahoo Finance (both free, no keys required). Then change `load_synthetic()` → `load()` in the dashboard.

---

## 📖 How It Works

### Data Pipeline
1. **Indicator Registry** (`indicators.py`) defines 30 indicators across FRED and Yahoo Finance
2. **CountryDataManager** (`country_manager.py`) organizes data by country with cross-country reference tables
3. In production: `fetch.py` pulls real data; in dev: synthetic data generated in-memory

### Feature Engineering
```
Base indicators (30 columns)
  → Cross-country lag correlations (+22 cols)
  → Global composites: GDP/Inflation/Risk indices (+5 cols)
  → Inter-sector spreads and ratios (+8 cols)
  → Time-series: lags, rolling mean/std, momentum, ROC (×5 per column)
  → Final: 200+ features
```

### Model Training
- For each target: train both **Random Forest** and **Gradient Boosting**
- 80/20 chronological split (no data leakage)
- Auto-select best model by RMSE
- Feature importances drive knowledge graph construction

### Simulation
1. Apply all scenario shocks simultaneously at T=0
2. Regenerate features → predict T+1 with best models → append results
3. Repeat for T+2, T+3 (cascading effects accumulate)
4. Compare baseline vs shocked trajectories with ±RMSE bounds

### XAI Explainability
- Trace causal paths through DAG from shock origins to each target
- Generate English narratives: *"Oil (+60%) cascades via [Oil → Shipping → EU Inflation]"*
- ±RMSE confidence bounds on every prediction

---

## 🗂️ Data Sources

| Source | API Key? | Indicators | Update Frequency |
|--------|----------|------------|------------------|
| **FRED** (Federal Reserve) | No | GDP, Unemployment, CPI, Interest Rates, Trade Balance | Monthly/Quarterly |
| **Yahoo Finance** | No | Commodities, Exchange Rates, Stock Indices, VIX | Daily |

---

## 🧪 Testing

```bash
# Test multi-country data architecture
python scripts/test_countries.py

# Test feature engineering pipeline
python scripts/test_features.py

# Test multi-model ML pipeline  
python scripts/test_ml_pipeline.py
```

---

## 📋 Project Milestones

### v1.0 ✅ (Completed)
Proof-of-concept with synthetic CSV data, 3 variables, Random Forest, basic DAG, Streamlit dashboard.

### v2.0 ✅ (Current)
- Real-world data integration (FRED + Yahoo Finance)
- 5 economies, 30+ indicators
- Multi-model ML pipeline with auto-selection
- Cross-country knowledge graph with structural edges
- 6 pre-built macro shock scenarios
- 4-tab premium dashboard

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ML | scikit-learn (Random Forest, Gradient Boosting) |
| Graph | NetworkX (in-memory DAG) |
| Dashboard | Streamlit + Plotly + Pyvis |
| Data APIs | FRED (public CSV) + yfinance |
| Data Format | Pandas DataFrames + CSV |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
