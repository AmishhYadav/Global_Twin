---
wave: 1
depends_on: []
files_modified: ["requirements.txt", "src/dashboard/app.py"]
autonomous: true
---

# Plan 1: Streamlit Visual Dashboard

## Objective
Construct the user interface `src/dashboard/app.py` allowing users to natively manipulate simulation inputs and dynamically view mathematical outputs generated structurally by Phase 1-5's logic stack.

## Tasks

<task>
<action>
Append `streamlit==1.32.2` and `plotly==5.20.0` to `requirements.txt`.
</action>
<read_first>
`requirements.txt`
</read_first>
<acceptance_criteria>
`cat requirements.txt | grep "streamlit"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `src/dashboard/` and build `src/dashboard/app.py`.
Logic:
1. Initialize Streamlit config (`st.set_page_config`).
2. Add `st.sidebar`: User selects Target Variable (e.g., `Oil_Price`) and Shock Magnitude slider (e.g., `-50%` to `+50%`).
3. Add `st.button` ("Run Simulation"): When clicked, trigger the full backend python stack mapping natively to:
   - Load Historical CSV dynamically.
   - Generate Features & Train Models.
   - Run Topology Engine & Cascade Simulation.
   - Run XAI formatter and parse the returned JSON string into Python dict.
4. Render Results: Loop the dictionary keys. For each Target node:
   - Instantiate `plotly.graph_objects.Figure()`.
   - Add Baseline and Shocked Scatter Lines natively.
   - Add `fill='tonexty'` logic dynamically embedding `bounds['lower_bound']` and `bounds['upper_bound']` generating visual standard deviations reliably.
   - Pass Figure into `st.plotly_chart()`.
   - Display `st.info(explanation)` string immediately tracking the DAG structurally below the render.
</action>
<read_first>
`src/xai/explainer.py`
</read_first>
<acceptance_criteria>
`cat src/dashboard/app.py | grep "st.plotly_chart"` returns a match.
</acceptance_criteria>
</task>

## Verification
- Running `streamlit run src/dashboard/app.py` successfully mounts the web server rendering the interface inputs perfectly bridging backend ML topologies reliably.

## Must Haves
- Python Streamlit native mapping embedding analytical Plotly objects safely.
- Explicit textual strings rendered natively natively mapping standard JSON architectures securely.
