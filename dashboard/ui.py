import plotly.graph_objects as go
import streamlit as st

from dashboard.constants import COURSE_MAP


def safe_course_name(code):
    try:
        return COURSE_MAP.get(int(code), f"Course {code}")
    except Exception:
        return str(code)


def gauge_chart(value, title, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%", "font": {"color": color, "size": 28}},
            title={"text": title, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def dark_layout(fig, height=None):
    # Keep backgrounds transparent so Plotly follows Streamlit light/dark theme.
    upd = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if height:
        upd["height"] = height
    fig.update_layout(**upd)
    return fig


def enforce_integer_year_axis(fig, axis="x"):
    axis_kwargs = {"tickmode": "linear", "tick0": 0, "dtick": 1, "tickformat": "d"}
    if axis in ("x", "both"):
        fig.update_xaxes(**axis_kwargs)
    if axis in ("y", "both"):
        fig.update_yaxes(**axis_kwargs)
    return fig


def kpi(col, val, label, delta="", color="#60a5fa"):
    col.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-value" style="color:{color}">{val}</p>'
        f'<p class="kpi-label">{label}</p>'
        f'<p class="kpi-delta">{delta}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def persistent_tab_selector(state_key, options, label="Section"):
    if not options:
        return None

    if state_key not in st.session_state or st.session_state[state_key] not in options:
        st.session_state[state_key] = options[0]

    return st.radio(
        label,
        options,
        key=state_key,
        horizontal=True,
        label_visibility="collapsed",
    )
