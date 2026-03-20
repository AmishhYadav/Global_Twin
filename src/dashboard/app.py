"""
Global Twin v2.0 — Dashboard
Clean, tech-friendly interface for economic simulation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.country_manager import CountryDataManager
from src.data.indicators import COUNTRIES, COUNTRY_INDICATORS, GLOBAL_INDICATORS
from src.features.build_features import create_time_series_features
from src.models.train import train_models
from src.graph.build_graph import create_knowledge_graph, get_graph_summary
from src.simulation.engine import run_simulation
from src.simulation.scenarios import SCENARIOS, list_scenarios, get_scenario
from src.xai.explainer import generate_xai_payload

# ── Page Config ──
st.set_page_config(page_title="Global Twin v2.0", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; border-radius: 8px;
        padding: 10px 20px; color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Backend Init (cached — runs once) ──
@st.cache_resource
def init_backend():
    import time
    t0 = time.time()
    mgr = CountryDataManager()
    mgr.load_synthetic()
    df = mgr.get_all_data()
    feat_df = create_time_series_features(df, lags=[1], rolling_windows=[5])
    print(f"  [Init] Features: {feat_df.shape[1]} cols ({time.time()-t0:.1f}s)")
    targets = ['CRUDE_OIL', 'SP500', 'GOLD', 'US_GDP_GROWTH', 'INR_USD']
    available = [t for t in targets if t in feat_df.columns]
    models = train_models(feat_df, available, verbose=False)
    print(f"  [Init] Models: {len(models)} ({time.time()-t0:.1f}s)")
    os.makedirs("models", exist_ok=True)
    G = create_knowledge_graph(models, importance_threshold=0.03, save_path="models/v2_graph.json")
    print(f"  [Init] Done in {time.time()-t0:.1f}s")
    return mgr, df, feat_df, models, G


# ── Helper: Render simulation results ──
def render_results(payload, horizon):
    """Show simulation charts with simple explanations."""
    nodes = payload.get("nodes", {})
    if not nodes:
        st.warning("No results.")
        return

    applied = payload.get("applied_shocks", {})
    if applied:
        with st.expander("🔧 What was changed (input shocks)", expanded=False):
            rows = [{"Variable": v, "Before": f"{i['original']:.2f}",
                      "After": f"{i['shocked']:.2f}", "Change": f"{i['pct']*100:+.0f}%"}
                    for v, i in applied.items()]
            st.table(pd.DataFrame(rows))

    for var, data in nodes.items():
        traj = data.get('trajectory', [])
        if not traj:
            continue

        st.subheader(f"{'🎯' if data.get('is_shock_origin') else '📡'} {var}")

        col_chart, col_info = st.columns([3, 1])

        with col_chart:
            steps = [f"T+{s['step']}" for s in traj]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps + steps[::-1],
                y=[s['bounds']['upper_bound'] for s in traj] +
                  [s['bounds']['lower_bound'] for s in traj][::-1],
                fill='toself', fillcolor='rgba(59,130,246,0.1)',
                line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(x=steps, y=[s['baseline'] for s in traj],
                mode='lines+markers', name='No Shock', line=dict(color='#6b7280', dash='dot')))
            fig.add_trace(go.Scatter(x=steps, y=[s['shocked'] for s in traj],
                mode='lines+markers', name='After Shock', line=dict(color='#3b82f6', width=3)))
            fig.update_layout(height=280, template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            final = traj[-1]
            direction = "📈" if final['percentage_delta'] > 0 else "📉"
            st.metric(f"{direction} T+{horizon} Change", f"{final['percentage_delta']:+.1f}%")
            tag = "Directly shocked" if data.get('is_shock_origin') else "Cascade effect"
            st.caption(tag)
            explanation = data.get('explanation', '')
            if explanation:
                st.info(explanation)

        st.divider()


# ── Load ──
st.title("🌍 Global Twin v2.0")
st.caption("Simulate how economic shocks ripple across countries — powered by ML")

with st.spinner("⚙️ Training ML models (first load only, ~7s)..."):
    mgr, base_df, feat_df, models, graph = init_backend()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Simulate", "🕸️ Graph", "ℹ️ How It Works"])


# ═══════════════════════════════════════
#  TAB 1: OVERVIEW — simple metrics + charts
# ═══════════════════════════════════════
with tab1:
    st.header("Global Snapshot")
    st.caption("Latest values from the simulation data (synthetic). Green = up from last week, Red = down.")

    latest = base_df.iloc[-1]
    week_ago = base_df.iloc[-7]

    def delta(curr, prev):
        if prev != 0:
            return f"{((curr - prev) / abs(prev)) * 100:+.1f}%"
        return "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🛢️ Oil Price", f"${latest['CRUDE_OIL']:.0f}", delta(latest['CRUDE_OIL'], week_ago['CRUDE_OIL']))
    c2.metric("📈 S&P 500", f"{latest['SP500']:.0f}", delta(latest['SP500'], week_ago['SP500']))
    c3.metric("🥇 Gold", f"${latest['GOLD']:.0f}", delta(latest['GOLD'], week_ago['GOLD']))
    c4.metric("😱 VIX (Fear)", f"{latest['VIX']:.1f}", delta(latest['VIX'], week_ago['VIX']), delta_color="inverse")
    c5.metric("💱 INR/USD", f"₹{latest['INR_USD']:.1f}", delta(latest['INR_USD'], week_ago['INR_USD']), delta_color="inverse")

    st.divider()

    # Two simple charts
    left, right = st.columns(2)

    with left:
        st.subheader("Commodity Prices (last year)")
        comm = ['CRUDE_OIL', 'GOLD', 'COPPER']
        avail = [c for c in comm if c in base_df.columns]
        if avail:
            norm = base_df[avail].tail(365)
            norm = norm / norm.iloc[0] * 100  # normalize to 100
            fig = px.line(norm, labels={"value": "Index (start=100)", "variable": ""})
            fig.update_layout(height=300, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 All prices normalized to 100 at start so you can compare % movement")

    with right:
        st.subheader("GDP Growth by Country")
        gdp = {'🇺🇸 US': 'US_GDP_GROWTH', '🇨🇳 China': 'CN_GDP_GROWTH',
               '🇮🇳 India': 'IN_GDP_GROWTH', '🇯🇵 Japan': 'JP_GDP_GROWTH'}
        avail_gdp = {k: v for k, v in gdp.items() if v in base_df.columns}
        if avail_gdp:
            vals = {k: base_df[v].iloc[-1] for k, v in avail_gdp.items()}
            fig = go.Figure(go.Bar(x=list(vals.keys()), y=list(vals.values()),
                marker_color=['#3b82f6', '#ef4444', '#f97316', '#14b8a6']))
            fig.update_layout(height=300, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10),
                yaxis_title="GDP Value")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 GDP = total economic output. Higher = stronger economy")


# ═══════════════════════════════════════
#  TAB 2: SIMULATE — run scenarios
# ═══════════════════════════════════════
with tab2:
    st.header("🔬 What-If Simulator")
    st.caption("Pick a scenario to see how economic shocks cascade through the system")

    mode = st.radio("", ["🎯 Pre-built Scenarios", "🔧 Custom Shock"], horizontal=True, label_visibility="collapsed")

    if mode == "🎯 Pre-built Scenarios":
        scenario_list = list_scenarios()

        # Simple cards
        for s in scenario_list:
            severity_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(s['severity'], "⚪")
            st.markdown(f"**{s['icon']} {s['name']}** {severity_color} — {s['description']}")

        st.divider()
        selected = st.selectbox("Choose scenario",
            [s['id'] for s in scenario_list],
            format_func=lambda x: f"{SCENARIOS[x]['icon']} {SCENARIOS[x]['name']}")

        scenario = get_scenario(selected)

        with st.expander(f"📋 What does '{scenario['name']}' do?", expanded=True):
            for var, pct in scenario['shocks'].items():
                direction = "⬆️" if pct > 0 else "⬇️"
                st.write(f"{direction} **{var}**: {pct*100:+.0f}%")

        horizon = st.slider("How many time steps to simulate?", 1, 5, 3)

        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner(f"Simulating {scenario['name']}..."):
                sim = run_simulation(models, base_df.tail(60),
                    shocks=scenario['shocks'], horizon=horizon, scenario_name=scenario['name'])
                payload = json.loads(generate_xai_payload(sim, models, graph, shocks=scenario['shocks']))
            st.success(f"✅ Done! Here's what happens:")
            render_results(payload, horizon)

    else:
        st.subheader("Build Your Own Shock")
        st.caption("Pick any variable and change its value to see the ripple effect")

        available_vars = sorted(list(models.keys()))
        num = st.number_input("How many variables to shock?", 1, 5, 1)
        custom_shocks = {}
        for i in range(int(num)):
            c1, c2 = st.columns([2, 1])
            with c1:
                var = st.selectbox(f"Variable {i+1}", available_vars, key=f"cv_{i}")
            with c2:
                pct = st.slider(f"Change %", -50, 100, 20, 5, key=f"cp_{i}")
            custom_shocks[var] = pct / 100.0

        horizon = st.slider("Time steps", 1, 5, 3, key="ch")

        if st.button("🚀 Run Custom Simulation", type="primary"):
            with st.spinner("Simulating..."):
                sim = run_simulation(models, base_df.tail(60),
                    shocks=custom_shocks, horizon=horizon, scenario_name="Custom")
                payload = json.loads(generate_xai_payload(sim, models, graph, shocks=custom_shocks))
            st.success("✅ Done!")
            render_results(payload, horizon)


# ═══════════════════════════════════════
#  TAB 3: GRAPH — knowledge graph
# ═══════════════════════════════════════
with tab3:
    st.header("🕸️ Causal Graph")
    st.caption("This graph shows which economic variables influence each other. "
               "Arrows mean 'A affects B'. Color = country, thickness = strength.")

    summary = get_graph_summary(graph)
    c1, c2, c3 = st.columns(3)
    c1.metric("Variables (nodes)", summary['total_nodes'])
    c2.metric("Connections (edges)", summary['total_edges'])
    c3.metric("Cross-border links", summary['cross_border_edges'])

    # Build graph
    net = Network(height="450px", width="100%", bgcolor="#0e1117", font_color="#e2e8f0", directed=True)
    net.force_atlas_2based()

    colors = {"US": "#3b82f6", "EU": "#8b5cf6", "CN": "#ef4444",
              "IN": "#f97316", "JP": "#14b8a6", "GLOBAL": "#6b7280"}

    for node in graph.nodes():
        country = graph.nodes[node].get('country', 'GLOBAL')
        sector = graph.nodes[node].get('sector', 'macro')
        net.add_node(node, label=node, color=colors.get(country, "#6b7280"),
            title=f"{node}\nCountry: {country}\nSector: {sector}", size=18)

    for u, v, d in graph.edges(data=True):
        etype = d.get('edge_type', 'ml')
        w = d.get('weight', 0.1)
        color = "#ef4444" if etype == "structural" else "#3b82f6" if etype == "hybrid" else "#4a4a6a"
        net.add_edge(u, v, value=w, title=f"{w:.3f} ({etype})", color=color)

    html_path = "/tmp/gt_graph.html"
    net.save_graph(html_path)
    with open(html_path, 'r') as f:
        components.html(f.read(), height=470)

    st.markdown("🔵 US  🟣 EU  🔴 China  🟠 India  🟢 Japan  ⚫ Global")
    st.caption("Red edges = known economic relationships (textbook). Blue = discovered by ML. Dark = ML-only.")


# ═══════════════════════════════════════
#  TAB 4: HOW IT WORKS
# ═══════════════════════════════════════
with tab4:
    st.header("ℹ️ How Global Twin Works")

    st.markdown("""
    ### The Big Idea
    Global Twin simulates **how economic shocks spread across countries** — like how an oil price spike
    in the Middle East can affect inflation in India and stock markets in the US.

    ### Pipeline (think of it like a data pipeline)

    ```
    Raw Data (30 indicators × 5 countries)
        ↓
    Feature Engineering (lags, rolling stats → 192 features)
        ↓
    ML Training (Random Forest per target variable)
        ↓
    Knowledge Graph (which variables predict which)
        ↓
    Simulation Engine (apply shock → re-predict → cascade)
        ↓
    XAI (trace causal path through graph → explain in English)
    ```

    ### Key Concepts (for non-finance people)

    | Term | What it means |
    |------|--------------|
    | **GDP** | Total economic output of a country. Think of it as "how much stuff a country produces" |
    | **CPI / Inflation** | How fast prices are rising. Higher = things get more expensive |
    | **Fed Rate** | The interest rate set by the US central bank. Higher = borrowing costs more |
    | **VIX** | The "fear index". Higher = more market uncertainty |
    | **S&P 500** | Index of 500 biggest US companies. Proxy for "how's the US stock market doing" |
    | **INR/USD** | How many Indian Rupees per 1 US Dollar. Higher = Rupee is weaker |
    | **Crude Oil** | Price of oil per barrel. Affects everything — transport, manufacturing, inflation |

    ### What does a "shock" do?
    When you shock a variable (e.g., Oil +60%), the system:
    1. Changes that variable's value
    2. Re-generates all features
    3. Runs every ML model to predict the next time step
    4. Repeats for T+2, T+3 (effects cascade through the graph)
    5. Compares "what would have happened normally" vs "what happens after the shock"

    ### Data
    Currently running on **synthetic data** (realistic fake data for development).
    Real data from FRED + Yahoo Finance is available in `data/raw/` — both free, no API keys needed.
    """)
