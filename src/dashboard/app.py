"""
Global Twin v2.0 — Multi-Tab Dashboard

Tabs:
  🌍 World View     — Global overview with key indicators and heatmap
  📊 Country Dive   — Per-country detailed charts  
  🔬 Scenario Lab   — Pre-built + custom shock simulation
  🕸️ Graph Explorer — Interactive cross-country knowledge graph
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
from src.features.build_features import build_full_feature_matrix, create_time_series_features
from src.models.train import train_models, get_comparison_report
from src.graph.build_graph import create_knowledge_graph, get_graph_summary
from src.simulation.engine import run_simulation
from src.simulation.scenarios import SCENARIOS, list_scenarios, get_scenario, build_custom_scenario
from src.xai.explainer import generate_xai_payload


# ─────────────────────────────────────────────
#  Page Config & Theme
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Global Twin v2.0",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2a4a;
        margin: 5px 0;
    }
    .scenario-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #2a3a5a;
        margin: 8px 0;
        cursor: pointer;
    }
    .scenario-card:hover { border-color: #e94560; }
    h1 { color: #e2e8f0 !important; }
    h2 { color: #cbd5e1 !important; }
    h3 { color: #94a3b8 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 10px 20px;
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e94560 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Data Loading & Model Training (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def init_backend():
    """Initialize data, features, models, and graph."""
    mgr = CountryDataManager()
    mgr.load_synthetic()
    
    df = mgr.get_all_data()
    feat_df = build_full_feature_matrix(df, lags=[1, 3], rolling_windows=[7])
    
    # Train models on key targets across countries
    targets = [
        'CRUDE_OIL', 'SP500', 'GOLD', 'VIX', 'BALTIC_DRY_INDEX',
        'US_CPI_INFLATION', 'US_UNEMPLOYMENT', 'US_GDP_GROWTH',
        'EU_CPI_INFLATION', 'EU_UNEMPLOYMENT',
        'CN_GDP_GROWTH', 'IN_GDP_GROWTH', 'JP_GDP_GROWTH',
        'EUR_USD', 'CNY_USD', 'INR_USD', 'JPY_USD',
    ]
    # Only train on targets that exist in feature matrix
    available_targets = [t for t in targets if t in feat_df.columns]
    
    models = train_models(feat_df, available_targets, verbose=False)
    
    os.makedirs("models", exist_ok=True)
    G = create_knowledge_graph(models, importance_threshold=0.03, save_path="models/v2_graph.json")
    
    return mgr, df, feat_df, models, G


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

st.title("🌍 Global Twin v2.0")
st.caption("Multi-country macroeconomic simulation engine with AI-powered causal analysis")

with st.spinner("⚙️ Initializing ML models across 5 economies..."):
    mgr, base_df, feat_df, models, graph = init_backend()


# ─────────────────────────────────────────────
#  Tab Layout
# ─────────────────────────────────────────────

tab_world, tab_country, tab_scenario, tab_graph = st.tabs([
    "🌍 World View", "📊 Country Dive", "🔬 Scenario Lab", "🕸️ Graph Explorer"
])


# ═══════════════════════════════════════════════
#  TAB 1: WORLD VIEW
# ═══════════════════════════════════════════════

with tab_world:
    st.header("Global Economic Snapshot")
    
    # Key global metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest = base_df.iloc[-1]
    prev = base_df.iloc[-7]
    
    def safe_delta(current, previous):
        if previous != 0:
            return f"{((current - previous) / abs(previous)) * 100:+.1f}%"
        return "N/A"
    
    with col1:
        st.metric("🛢️ Crude Oil", f"${latest['CRUDE_OIL']:.1f}", safe_delta(latest['CRUDE_OIL'], prev['CRUDE_OIL']))
    with col2:
        st.metric("📈 S&P 500", f"{latest['SP500']:.0f}", safe_delta(latest['SP500'], prev['SP500']))
    with col3:
        st.metric("🥇 Gold", f"${latest['GOLD']:.0f}", safe_delta(latest['GOLD'], prev['GOLD']))
    with col4:
        st.metric("😱 VIX", f"{latest['VIX']:.1f}", safe_delta(latest['VIX'], prev['VIX']), delta_color="inverse")
    with col5:
        st.metric("🚢 Baltic Dry", f"{latest['BALTIC_DRY_INDEX']:.0f}", safe_delta(latest['BALTIC_DRY_INDEX'], prev['BALTIC_DRY_INDEX']))
    
    st.markdown("---")
    
    # Commodity trends
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Commodity Prices (Normalized)")
        commodity_cols = ['CRUDE_OIL', 'NATURAL_GAS', 'GOLD', 'COPPER', 'WHEAT']
        available_comm = [c for c in commodity_cols if c in base_df.columns]
        if available_comm:
            norm_df = base_df[available_comm].tail(365)
            norm_df = norm_df / norm_df.iloc[0] * 100
            fig = px.line(norm_df, labels={"value": "Index (Base=100)", "DATE": "Date"})
            fig.update_layout(
                height=350, template="plotly_dark",
                margin=dict(l=20, r=20, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Exchange Rates vs USD")
        fx_cols = ['EUR_USD', 'CNY_USD', 'INR_USD', 'JPY_USD']
        available_fx = [c for c in fx_cols if c in base_df.columns]
        if available_fx:
            fx_norm = base_df[available_fx].tail(365)
            fx_norm = fx_norm / fx_norm.iloc[0] * 100
            fig = px.line(fx_norm, labels={"value": "Index (Base=100)", "DATE": "Date"})
            fig.update_layout(
                height=350, template="plotly_dark",
                margin=dict(l=20, r=20, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Country GDP comparison
    st.subheader("GDP Growth by Country")
    gdp_cols = {
        '🇺🇸 US': 'US_GDP_GROWTH', '🇪🇺 EU': 'EU_GDP_GROWTH',
        '🇨🇳 China': 'CN_GDP_GROWTH', '🇮🇳 India': 'IN_GDP_GROWTH', '🇯🇵 Japan': 'JP_GDP_GROWTH'
    }
    available_gdp = {k: v for k, v in gdp_cols.items() if v in base_df.columns}
    if available_gdp:
        gdp_latest = {k: base_df[v].iloc[-1] for k, v in available_gdp.items()}
        fig = go.Figure(go.Bar(
            x=list(gdp_latest.keys()), y=list(gdp_latest.values()),
            marker_color=['#3b82f6', '#8b5cf6', '#ef4444', '#f97316', '#14b8a6']
        ))
        fig.update_layout(
            height=300, template="plotly_dark",
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis_title="GDP Value"
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
#  TAB 2: COUNTRY DEEP-DIVE
# ═══════════════════════════════════════════════

with tab_country:
    st.header("Country Deep-Dive")
    
    selected_country = st.selectbox(
        "Select Economy",
        list(COUNTRIES.keys()),
        format_func=lambda x: f"{COUNTRIES[x]} ({x})"
    )
    
    country_df = mgr.get_country_data(selected_country)
    country_indicators = COUNTRY_INDICATORS.get(selected_country, [])
    
    # Country-specific indicators
    st.subheader(f"{COUNTRIES[selected_country]} — Key Indicators")
    
    country_specific = [c for c in country_indicators if c in country_df.columns]
    
    if country_specific:
        # Metrics row
        metric_cols = st.columns(min(len(country_specific), 4))
        for i, ind in enumerate(country_specific[:4]):
            with metric_cols[i]:
                val = country_df[ind].iloc[-1]
                prev_val = country_df[ind].iloc[-30] if len(country_df) > 30 else val
                st.metric(ind.split('_', 1)[-1].replace('_', ' ').title(), f"{val:.2f}", safe_delta(val, prev_val))
        
        # Time series charts
        st.subheader("Historical Trends")
        selected_indicators = st.multiselect(
            "Select indicators to chart",
            country_specific,
            default=country_specific[:2]
        )
        
        if selected_indicators:
            chart_period = st.slider("Days of history", 30, len(country_df), 365, key="country_days")
            chart_data = country_df[selected_indicators].tail(chart_period)
            
            for ind in selected_indicators:
                fig = px.line(chart_data, y=ind, title=ind)
                fig.update_layout(height=300, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
    
    # Global indicators available to this country
    st.subheader("Global Market Indicators")
    global_in_country = [c for c in GLOBAL_INDICATORS if c in country_df.columns]
    if global_in_country:
        sel_global = st.multiselect("Select global indicators", global_in_country, default=global_in_country[:3])
        if sel_global:
            norm_global = country_df[sel_global].tail(365)
            norm_global = norm_global / norm_global.iloc[0] * 100
            fig = px.line(norm_global, labels={"value": "Index (Base=100)"})
            fig.update_layout(height=350, template="plotly_dark", margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
#  TAB 3: SCENARIO LAB
# ═══════════════════════════════════════════════

with tab_scenario:
    st.header("🔬 Scenario Lab")
    st.markdown("Simulate pre-built macroeconomic scenarios or build your own custom shock.")
    
    mode = st.radio("Mode", ["Pre-Built Scenarios", "Custom Shock Builder"], horizontal=True)
    
    if mode == "Pre-Built Scenarios":
        scenario_list = list_scenarios()
        
        # Scenario cards
        cols = st.columns(3)
        for i, s in enumerate(scenario_list):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"### {s['icon']} {s['name']}")
                    st.caption(f"Severity: **{s['severity'].upper()}** · {s['num_shocks']} variables")
                    st.write(s['description'])
        
        selected_scenario = st.selectbox(
            "Select Scenario to Execute",
            [s['id'] for s in scenario_list],
            format_func=lambda x: f"{SCENARIOS[x]['icon']} {SCENARIOS[x]['name']}"
        )
        
        scenario = get_scenario(selected_scenario)
        
        # Show shocks
        with st.expander("📋 Shock Details", expanded=True):
            shock_data = []
            for var, pct in scenario['shocks'].items():
                shock_data.append({"Variable": var, "Shock": f"{pct*100:+.0f}%"})
            st.table(pd.DataFrame(shock_data))
        
        horizon = st.slider("Forecast Horizon (T+N)", 1, 5, 3, key="scenario_horizon")
        run_scenario = st.button("🚀 Execute Scenario", type="primary", key="run_scenario")
        
        if run_scenario:
            with st.spinner(f"Running {scenario['name']} across T+{horizon}..."):
                sim = run_simulation(
                    models_dict=models,
                    base_df=base_df.tail(60),
                    shocks=scenario['shocks'],
                    horizon=horizon,
                    scenario_name=scenario['name'],
                )
                payload = json.loads(generate_xai_payload(sim, models, graph, shocks=scenario['shocks']))
                
                st.success(f"✅ {scenario['name']} simulation complete!")
                _render_simulation_results(payload, horizon)
    
    else:
        # Custom builder
        st.subheader("Build Custom Scenario")
        
        available_vars = sorted(list(models.keys()))
        custom_shocks = {}
        
        num_shocks = st.number_input("Number of variables to shock", 1, 10, 2)
        
        for i in range(int(num_shocks)):
            col_var, col_pct = st.columns([2, 1])
            with col_var:
                var = st.selectbox(f"Variable {i+1}", available_vars, key=f"custom_var_{i}")
            with col_pct:
                pct = st.slider(f"Shock %", -80, 150, 20, 5, key=f"custom_pct_{i}")
            custom_shocks[var] = pct / 100.0
        
        horizon = st.slider("Forecast Horizon", 1, 5, 3, key="custom_horizon")
        run_custom = st.button("🚀 Execute Custom Scenario", type="primary", key="run_custom")
        
        if run_custom:
            with st.spinner("Running custom simulation..."):
                sim = run_simulation(
                    models_dict=models,
                    base_df=base_df.tail(60),
                    shocks=custom_shocks,
                    horizon=horizon,
                    scenario_name="Custom Scenario",
                )
                payload = json.loads(generate_xai_payload(sim, models, graph, shocks=custom_shocks))
                
                st.success("✅ Custom simulation complete!")
                _render_simulation_results(payload, horizon)


# ═══════════════════════════════════════════════
#  TAB 4: GRAPH EXPLORER
# ═══════════════════════════════════════════════

with tab_graph:
    st.header("🕸️ Knowledge Graph Explorer")
    
    summary = get_graph_summary(graph)
    
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Nodes", summary['total_nodes'])
    col_g2.metric("Edges", summary['total_edges'])
    col_g3.metric("Cross-Border Edges", summary['cross_border_edges'])
    col_g4.metric("Is DAG", "✓" if summary['is_dag'] else "✗")
    
    # Build Pyvis graph
    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="#e2e8f0", directed=True)
    net.force_atlas_2based()
    
    # Color nodes by country
    country_colors = {
        "US": "#3b82f6", "EU": "#8b5cf6", "CN": "#ef4444",
        "IN": "#f97316", "JP": "#14b8a6", "GLOBAL": "#6b7280"
    }
    
    for node in graph.nodes():
        country = graph.nodes[node].get('country', 'GLOBAL')
        sector = graph.nodes[node].get('sector', 'macro')
        color = country_colors.get(country, "#6b7280")
        net.add_node(node, label=node, color=color, title=f"{node}\nCountry: {country}\nSector: {sector}", size=20)
    
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('edge_type', 'ml_derived')
        weight = data.get('weight', 0.1)
        color = "#e94560" if edge_type == "structural" else "#4a9eff" if edge_type == "hybrid" else "#3a3a5a"
        net.add_edge(u, v, value=weight, title=f"Weight: {weight:.3f}\nType: {edge_type}", color=color)
    
    html_path = "/tmp/v2_knowledge_graph.html"
    net.save_graph(html_path)
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    components.html(html_data, height=520)
    
    # Legend
    st.markdown("**Node Colors:** 🔵 US · 🟣 EU · 🔴 China · 🟠 India · 🟢 Japan · ⚫ Global")
    st.markdown("**Edge Colors:** 🔴 Structural · 🔵 Hybrid · ⬛ ML-Derived")
    
    # Node/edge details
    with st.expander("📋 Graph Details"):
        for country, nodes in summary['countries'].items():
            st.write(f"**{country}**: {', '.join(sorted(nodes))}")


# ─────────────────────────────────────────────
#  Shared: Render Simulation Results
# ─────────────────────────────────────────────

def _render_simulation_results(payload, horizon):
    """Render interactive charts and XAI narratives."""
    
    nodes = payload.get("nodes", {})
    
    if not nodes:
        st.warning("No results to display.")
        return
    
    # Applied shocks summary
    applied = payload.get("applied_shocks", {})
    if applied:
        with st.expander("📊 Applied Shocks", expanded=False):
            shock_rows = []
            for var, info in applied.items():
                shock_rows.append({
                    "Variable": var,
                    "Original": f"{info['original']:.2f}",
                    "Shocked": f"{info['shocked']:.2f}",
                    "Change": f"{info['pct']*100:+.0f}%",
                })
            st.table(pd.DataFrame(shock_rows))
    
    # Per-variable results
    for target_var, data in nodes.items():
        traj = data.get('trajectory', [])
        if not traj:
            continue
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 {target_var}")
            
            x_steps = [f"T+{s['step']}" for s in traj]
            y_base = [s['baseline'] for s in traj]
            y_shock = [s['shocked'] for s in traj]
            y_lower = [s['bounds']['lower_bound'] for s in traj]
            y_upper = [s['bounds']['upper_bound'] for s in traj]
            
            fig = go.Figure()
            
            # Confidence band
            fig.add_trace(go.Scatter(
                x=x_steps + x_steps[::-1], y=y_upper + y_lower[::-1],
                fill='toself', fillcolor='rgba(233,69,96,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='±RMSE'
            ))
            
            fig.add_trace(go.Scatter(
                x=x_steps, y=y_base, mode='lines+markers',
                name='Baseline', line=dict(color='#6b7280', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=x_steps, y=y_shock, mode='lines+markers',
                name='Shocked', line=dict(color='#e94560', width=3)
            ))
            
            fig.update_layout(
                height=350, template="plotly_dark",
                margin=dict(l=20, r=20, t=10, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            final = traj[-1]
            st.metric(
                f"T+{horizon} Impact",
                f"{final['percentage_delta']:+.2f}%",
                delta=f"{final['absolute_delta']:+.2f}"
            )
            
            origin_tag = "🎯 SHOCKED" if data.get('is_shock_origin') else "📡 Cascaded"
            st.caption(origin_tag)
            
            st.info(f"**XAI:**\n\n{data.get('explanation', 'N/A')}")
        
        st.markdown("---")
