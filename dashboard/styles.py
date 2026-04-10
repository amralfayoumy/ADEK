import streamlit as st


GLOBAL_STYLES = """
<style>
[data-testid="stAppViewContainer"] { background: var(--background-color); }
[data-testid="stSidebar"]          { background: var(--secondary-background-color); border-right: 1px solid rgba(120, 120, 120, 0.25); }
h1,h2,h3,h4 { color: var(--text-color); }
p, li, label { color: var(--text-color); }

.kpi-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-value { font-size: 2.4rem; font-weight: 700; color: #60a5fa; margin: 0; }
.kpi-label { font-size: 0.85rem; color: var(--text-color); opacity: 0.75; margin: 4px 0 0; }
.kpi-delta { font-size: 0.8rem; margin-top: 4px; color: var(--text-color); opacity: 0.72; }

.badge-high   { background:#3d1212; color:#f87171; border:1px solid #f87171;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}
.badge-medium { background:#3d2d0a; color:#fbbf24; border:1px solid #fbbf24;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}
.badge-low    { background:#0d2b1a; color:#34d399; border:1px solid #34d399;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}

.section-header {
    font-size:1.15rem; font-weight:700; color:var(--text-color);
    border-left:3px solid var(--primary-color); padding-left:10px; margin:24px 0 12px;
}

.intervention-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.intervention-title { font-weight:700; color:var(--text-color); }

[data-testid="stDataFrame"] { border-radius: 8px; }
.stProgress > div > div { background: var(--primary-color); }

.uni-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}

/* Keep Sankey labels crisp in light mode (no ghost/shadow under text). */
.js-plotly-plot .sankey .node text {
    text-shadow: none !important;
    filter: none !important;
}
</style>
"""


def inject_global_styles():
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)
