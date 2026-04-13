import re

import plotly.graph_objects as go
import streamlit as st

from dashboard.constants import COURSE_MAP


_SEMESTER_REPLACEMENTS = (
    (re.compile(r"\b1st\s*semester\b", flags=re.IGNORECASE), "Fall Semester"),
    (re.compile(r"\b1st\s*sem\b", flags=re.IGNORECASE), "Fall Semester"),
    (re.compile(r"\b2nd\s*semester\b", flags=re.IGNORECASE), "Spring Semester"),
    (re.compile(r"\b2nd\s*sem\b", flags=re.IGNORECASE), "Spring Semester"),
)


def semesterized_text(text):
    if not isinstance(text, str):
        return text

    updated = text
    for pattern, replacement in _SEMESTER_REPLACEMENTS:
        updated = pattern.sub(replacement, updated)
    return updated


def semesterized_columns(df):
    if not hasattr(df, "columns") or not hasattr(df, "rename"):
        return df

    rename_map = {}
    for col in df.columns:
        if isinstance(col, str):
            repl = semesterized_text(col)
            if repl != col:
                rename_map[col] = repl

    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _semesterize_plotly_text(fig):
    if fig is None:
        return fig

    if getattr(fig.layout, "title", None) and isinstance(fig.layout.title.text, str):
        fig.layout.title.text = semesterized_text(fig.layout.title.text)

    if getattr(fig.layout, "legend", None) and getattr(fig.layout.legend, "title", None):
        if isinstance(fig.layout.legend.title.text, str):
            fig.layout.legend.title.text = semesterized_text(fig.layout.legend.title.text)

    if getattr(fig.layout, "annotations", None):
        for ann in fig.layout.annotations:
            if isinstance(getattr(ann, "text", None), str):
                ann.text = semesterized_text(ann.text)

    for key in fig.layout:
        if key.startswith(("xaxis", "yaxis", "coloraxis")):
            axis_obj = fig.layout[key]
            if getattr(axis_obj, "title", None) and isinstance(axis_obj.title.text, str):
                axis_obj.title.text = semesterized_text(axis_obj.title.text)

    def _semesterize_seq(values):
        if isinstance(values, (list, tuple)):
            return [semesterized_text(item) if isinstance(item, str) else item for item in values]
        if hasattr(values, "tolist") and not isinstance(values, (str, bytes)):
            listed = values.tolist()
            if isinstance(listed, list):
                return [semesterized_text(item) if isinstance(item, str) else item for item in listed]
        return values

    for trace in fig.data:
        if isinstance(getattr(trace, "name", None), str):
            trace.name = semesterized_text(trace.name)

        if isinstance(getattr(trace, "hovertemplate", None), str):
            trace.hovertemplate = semesterized_text(trace.hovertemplate)

        if isinstance(getattr(trace, "text", None), str):
            trace.text = semesterized_text(trace.text)

        if isinstance(getattr(trace, "text", None), (list, tuple)):
            trace.text = [semesterized_text(item) if isinstance(item, str) else item for item in trace.text]

        if getattr(trace, "x", None) is not None:
            trace.x = _semesterize_seq(trace.x)
        if getattr(trace, "y", None) is not None:
            trace.y = _semesterize_seq(trace.y)
        if getattr(trace, "labels", None) is not None:
            trace.labels = _semesterize_seq(trace.labels)

        if getattr(trace, "colorbar", None) and getattr(trace.colorbar, "title", None):
            if isinstance(trace.colorbar.title.text, str):
                trace.colorbar.title.text = semesterized_text(trace.colorbar.title.text)

    return fig


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
    _semesterize_plotly_text(fig)
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
        f'<p class="kpi-label">{semesterized_text(label)}</p>'
        f'<p class="kpi-delta">{delta}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
