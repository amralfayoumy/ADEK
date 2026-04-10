import streamlit as st


GLOBAL_STYLES = """
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a3040; }
h1,h2,h3,h4 { color: #e8eaf0; }
p, li, label { color: #b0b8c8; }

.kpi-card {
    background: linear-gradient(135deg, #1a2035, #1e2a45);
    border: 1px solid #2a3a5e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-value { font-size: 2.4rem; font-weight: 700; color: #60a5fa; margin: 0; }
.kpi-label { font-size: 0.85rem; color: #8090a8; margin: 4px 0 0; }
.kpi-delta { font-size: 0.8rem; margin-top: 4px; }

.badge-high   { background:#3d1212; color:#f87171; border:1px solid #f87171;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}
.badge-medium { background:#3d2d0a; color:#fbbf24; border:1px solid #fbbf24;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}
.badge-low    { background:#0d2b1a; color:#34d399; border:1px solid #34d399;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}

.section-header {
    font-size:1.15rem; font-weight:700; color:#93c5fd;
    border-left:3px solid #3b82f6; padding-left:10px; margin:24px 0 12px;
}

.intervention-card {
    background: #131c2e;
    border: 1px solid #2a3f6f;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.intervention-title { font-weight:700; color:#93c5fd; }

[data-testid="stDataFrame"] { border-radius: 8px; }
.stProgress > div > div { background: #3b82f6; }

.uni-card {
    background: linear-gradient(135deg, #1a2035, #1a2a3a);
    border: 1px solid #2a3a5e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
</style>
"""


def inject_global_styles():
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)
