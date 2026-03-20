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

    # ── CHAIN REACTION SUMMARY (the key addition) ──
    chain = payload.get("chain_reaction", [])
    if chain:
        st.subheader("⛓️ Chain Reaction — What happened and WHY")
        st.caption("Read top to bottom: your shock (⚡) triggers cascade effects (📡)")

        for item in chain:
            icon = "⚡" if item['type'] == 'direct_shock' else "📡"
            name = item.get('name', item['variable'])
            change = item['change']
            reason = item['reason']

            # Color the change
            color = "🟢" if change.startswith("+") else "🔴"

            st.markdown(f"{icon} **{name}** `{change}` {color}")
            st.caption(f"   ↳ {reason}")

            if item.get('path'):
                st.caption(f"   📍 Path: {item['path']}")

        st.divider()

    # ── PER-VARIABLE CHARTS ──
    for var, data in nodes.items():
        traj = data.get('trajectory', [])
        if not traj:
            continue

        var_name = data.get('var_name', var)
        is_origin = data.get('is_shock_origin', False)
        final = traj[-1]

        # Skip variables with negligible change
        if abs(final['percentage_delta']) < 0.01 and not is_origin:
            continue

        st.subheader(f"{'⚡' if is_origin else '📡'} {var_name} ({var})")

        col_chart, col_info = st.columns([3, 1])

        with col_chart:
            steps = [f"T+{s['step']}" for s in traj]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps + steps[::-1],
                y=[s['bounds']['upper_bound'] for s in traj] +
                  [s['bounds']['lower_bound'] for s in traj][::-1],
                fill='toself', fillcolor='rgba(59,130,246,0.1)',
                line=dict(color='rgba(0,0,0,0)'), name='Confidence', hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(x=steps, y=[s['baseline'] for s in traj],
                mode='lines+markers', name='Without shock', line=dict(color='#6b7280', dash='dot')))
            fig.add_trace(go.Scatter(x=steps, y=[s['shocked'] for s in traj],
                mode='lines+markers', name='With shock', line=dict(color='#3b82f6', width=3)))
            fig.update_layout(height=250, template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            direction = "📈" if final['percentage_delta'] > 0 else "📉"
            st.metric(f"{direction} Impact by T+{horizon}",
                      f"{final['percentage_delta']:+.1f}%")
            if is_origin:
                st.caption("⚡ You directly changed this")
            else:
                st.caption("📡 Changed due to ripple effect")

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
    st.markdown("""
    **How it works:** You change a variable (e.g. "Oil +60%"). The ML model then predicts what happens next:
    - **T+1** = tomorrow's predicted value
    - **T+2** = day after tomorrow
    - **T+3** = 3 days out
    
    The dotted line shows "no shock" (normal). The solid blue line shows the shock's effect.
    """)

    mode = st.radio("Simulation mode", ["🎯 Pre-built Scenarios", "🔧 Custom Shock"], horizontal=True, label_visibility="collapsed")

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

        horizon = st.slider("How many days ahead to predict (T+N)", 1, 5, 3)

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

        horizon = st.slider("Days ahead to predict (T+N)", 1, 5, 3, key="ch")

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

    st.markdown("### 🎯 What does the ML actually predict?")
    st.markdown("""
    Think of it like weather forecasting, but for economics.
    
    We have **5 target variables** the ML is trained to predict:
    - `CRUDE_OIL` — tomorrow's oil price
    - `SP500` — tomorrow's US stock market value
    - `GOLD` — tomorrow's gold price
    - `US_GDP_GROWTH` — tomorrow's US economic growth indicator
    - `INR_USD` — tomorrow's INR/USD exchange rate
    
    **How?** A Random Forest model looks at **192 features** (yesterday's values, rolling averages, 
    momentum, etc.) and learns patterns like:
    > *"When oil went up yesterday AND VIX was high, gold tends to go up tomorrow"*
    """)

    st.markdown("### ⏱️ What is T+1, T+2, T+3?")
    st.markdown("""
    `T` = "Today". `T+N` = "N days from now".
    
    **Concrete example — Oil Embargo scenario:**
    
    | Step | What happens |
    |------|-------------|
    | **T+0** | You shock Oil to +60%. Everything else stays the same. |
    | **T+1** | ML predicts: "Given oil at +60%, Gold will be X, S&P will be Y..." |
    | **T+2** | ML predicts again, but now using T+1's predicted values as input |
    | **T+3** | Same — now the cascade has compounded over 3 steps |
    
    This is why effects **grow over time** — each step feeds into the next, like dominoes.
    """)

    st.markdown("### 🔧 The Full Pipeline")
    st.code("""
    1. DATA:     30 economic indicators across US, EU, China, India, Japan
    2. FEATURES: Yesterday's values + rolling averages + momentum → 192 inputs
    3. TRAINING: Random Forest learns "given these 192 inputs, predict tomorrow"
    4. GRAPH:    Feature importances reveal which variables drive which
    5. SHOCK:    User changes a variable → ML re-predicts with new value
    6. CASCADE:  Repeat prediction using shocked outputs as next input
    7. EXPLAIN:  Trace the graph to explain WHY each variable changed
    """, language=None)

    st.markdown("### 📖 Finance Glossary")
    st.markdown("""
    | Term | Plain English |
    |------|--------------|
    | **GDP** | How much stuff a country produces. Higher = stronger economy |
    | **CPI / Inflation** | How fast prices rise. 2% = normal, 8% = bad |
    | **Fed Rate** | US interest rate. Higher = expensive to borrow money |
    | **VIX** | Fear index. Low (~15) = calm, High (~40) = panic |
    | **S&P 500** | Top 500 US companies' stock value. Proxy for "US market" |
    | **INR/USD** | Rupees per Dollar. ₹83 means $1 buys 83 rupees |
    | **Crude Oil** | Oil price per barrel (~$70-80 normal). Affects EVERYTHING |
    """)

    st.markdown("### 📊 Reading the Charts")
    st.markdown("""
    In the Simulate tab, each chart shows:
    - **Dotted gray line** = "What would happen normally" (no shock)
    - **Solid blue line** = "What happens after your shock"
    - **Light blue band** = Confidence range (model could be off by this much)
    - **% Change** on the right = How much the shock moved this variable
    - 🎯 = You directly changed this variable
    - 📡 = This variable changed because of cascade effects
    """)

    st.divider()
    st.caption("Currently running on synthetic data. Real data from FRED + Yahoo Finance is in data/raw/.")

