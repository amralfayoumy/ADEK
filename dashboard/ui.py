import plotly.graph_objects as go

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
            title={"text": title, "font": {"color": "#b0b8c8", "size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#b0b8c8"}},
                "bar": {"color": color},
                "bgcolor": "#1a2035",
                "bordercolor": "#2a3040",
                "steps": [
                    {"range": [0, 30], "color": "#0d2b1a"},
                    {"range": [30, 60], "color": "#3d2d0a"},
                    {"range": [60, 100], "color": "#3d1212"},
                ],
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f1117",
        font_color="#b0b8c8",
    )
    return fig


def dark_layout(fig, height=None):
    upd = dict(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font_color="#b0b8c8")
    if height:
        upd["height"] = height
    fig.update_layout(**upd)
    return fig


def kpi(col, val, label, delta="", color="#60a5fa"):
    col.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-value" style="color:{color}">{val}</p>'
        f'<p class="kpi-label">{label}</p>'
        f'<p class="kpi-delta" style="color:#8090a8">{delta}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
