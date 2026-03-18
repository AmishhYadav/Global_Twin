---
wave: 1
depends_on: []
files_modified: ["requirements.txt", "src/dashboard/app.py"]
autonomous: true
---

# Plan 1: Topology Visualizer Component

## Objective
Update `src/dashboard/app.py` extending the UI architecture natively drawing explicit physics boundaries mapping Knowledge Graph nodes.

## Tasks

<task>
<action>
Append `pyvis==0.3.2` to `requirements.txt`.
</action>
<read_first>
`requirements.txt`
</read_first>
<acceptance_criteria>
`cat requirements.txt | grep "pyvis"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Modify `src/dashboard/app.py`.
Logic:
1. Include `from pyvis.network import Network` and `import streamlit.components.v1 as components`.
2. Locate the existing `st.title` instantiation block. Immediately beneath it, extract the generated `graph` NetworkX topology logic.
3. Configure `pyvis.network.Network` (Setting layout variables like `width="100%"`, `height="350px"`).
4. Run `net.from_nx(graph)`.
5. Run `net.save_graph("knowledge_graph.html")`.
6. Read the newly injected structural file string `html_data`.
7. Bind the extracted HTML array directly to the HTTP environment via `components.html(html_data, height=360)`.
</action>
<read_first>
`src/dashboard/app.py`
</read_first>
<acceptance_criteria>
`cat src/dashboard/app.py | grep "import streamlit.components"` returns a match.
</acceptance_criteria>
</task>

## Verification
- Running `streamlit run src/dashboard/app.py` dynamically embeds a bouncy graphical physics mapping explicitly drawing the Target Nodes dependencies exactly.

## Must Haves
- Python Streamlit HTML component explicit mappings explicitly.
- Safely linking the `load_and_train_backend` dynamically instantiated Graph Object straight into GUI parameters securely.
