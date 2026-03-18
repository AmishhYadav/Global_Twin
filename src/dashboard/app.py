import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pyvis.network import Network

# Ensure src module is visible for nested imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.build_features import create_time_series_features
from src.models.train import train_models
from src.graph.build_graph import create_knowledge_graph
from src.simulation.engine import run_simulation
from src.xai.explainer import generate_xai_payload

st.set_page_config(page_title="Global Twin Simulator", layout="wide")

@st.cache_data
def load_and_train_backend():
    """Loads synthetic historical data, engineers features, trains models, and builds the knowledge graph."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    oil = np.random.normal(60, 5, len(dates))
    shipping = (oil * 5) + np.random.normal(100, 50, len(dates))
    inflation = (shipping * 0.01) + np.random.normal(2.0, 0.1, len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'Oil_Price': oil,
        'Shipping_Cost': shipping,
        'Inflation_Rate': inflation
    })
    
    # Set Date as index for feature engineering and training
    df_indexed = df.set_index('Date')
    
    feat_df = create_time_series_features(df_indexed, lags=[1, 3], rolling_windows=[7])
    target_cols = ['Shipping_Cost', 'Inflation_Rate']
    results_dict = train_models(feat_df, target_cols)
    
    os.makedirs("models", exist_ok=True)
    G = create_knowledge_graph(results_dict, importance_threshold=0.05, save_path="models/ui_graph.json")
    
    # Return the original df (with Date column) for display, models, and graph
    return df, results_dict, G

st.title("🌍 Global Twin: Supply Chain Simulator")
st.markdown("Inject percentage shocks into economic variables and observe downstream cascading effects predicted by Random Forest models across a Knowledge Graph.")

with st.spinner("Initializing ML Models and loading historical data..."):
    base_df, models, graph = load_and_train_backend()

# --- TOPOLOGY KNOWLEDGE GRAPH ---
with st.expander("📊 View Interactive Causal Topology Graph", expanded=True):
    net = Network(height="350px", width="100%", bgcolor="white", font_color="black", directed=True)
    net.force_atlas_2based()
    net.from_nx(graph)
    
    html_path = "/tmp/ui_knowledge_graph.html"
    net.save_graph(html_path)
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    components.html(html_data, height=365)

# --- Sidebar Inputs ---
st.sidebar.header("🎛️ Simulator Configuration")
shock_node = st.sidebar.selectbox("Target Origin Variable", ["Oil_Price", "Shipping_Cost", "Inflation_Rate"])

shock_pct_input = st.sidebar.slider("Shock Magnitude (%)", min_value=-50, max_value=50, value=20, step=5)
shock_pct_decimal = shock_pct_input / 100.0

run_sim = st.sidebar.button("🚀 Execute 'What-If' Simulation")

# --- Dashboard Main View ---
if run_sim:
    with st.spinner(f"Cascading {shock_pct_input:+d}% shock on {shock_node} across T+3 horizon..."):
        # Run simulation (pass the raw df — engine handles Date column)
        sim_out = run_simulation(
            models_dict=models,
            base_df=base_df.tail(60),
            shock_node=shock_node,
            shock_pct=shock_pct_decimal,
            horizon=3
        )
        payload_str = generate_xai_payload(sim_out, models, graph, shock_node)
        payload = json.loads(payload_str)
        
        st.success("✅ Simulation Complete!")
        
        # Parse output payload and build individual Node charts
        nodes_dict = payload.get("nodes", {})
        
        for idx, (target_var, data) in enumerate(nodes_dict.items()):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📈 Trajectory: {target_var}")
                fig = go.Figure()
                
                traj_array = data['trajectory']
                x_steps = [f"T+{step['step']}" for step in traj_array]
                y_base = [step['baseline'] for step in traj_array]
                y_shock = [step['shocked'] for step in traj_array]
                y_lower = [step['bounds']['lower_bound'] for step in traj_array]
                y_upper = [step['bounds']['upper_bound'] for step in traj_array]
                
                # Confidence band (±RMSE)
                fig.add_trace(go.Scatter(
                    x=x_steps + x_steps[::-1],
                    y=y_upper + y_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(231,107,243,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='± RMSE Confidence'
                ))
                
                # Baseline trace
                fig.add_trace(go.Scatter(
                    x=x_steps, y=y_base,
                    mode='lines+markers',
                    name='Baseline Forecast',
                    line=dict(color='gray', dash='dash')
                ))
                
                # Shocked trace
                fig.add_trace(go.Scatter(
                    x=x_steps, y=y_shock,
                    mode='lines+markers',
                    name=f'Shocked Forecast ({shock_pct_input:+d}%)',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.info(f"**XAI Explanation:**\n\n{data['explanation']}")
                
                final_delta = traj_array[-1]['percentage_delta']
                st.metric(label="Cumulative T+3 Impact", value=f"{final_delta:+.2f}%", delta=f"{final_delta:+.2f}%")
            
            st.markdown("---")
else:
    st.info("👈 Use the sidebar to select a variable, set a shock percentage, and click **Execute** to run the simulation.")
