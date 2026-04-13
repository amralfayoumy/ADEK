import json

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


def persist_streamlit_tabs(page_key, tab_labels):
    if not tab_labels:
        return

    payload = json.dumps({"page_key": page_key, "labels": tab_labels})
    st.iframe(
        f"""
<script>
(() => {{
    const cfg = {payload};
    const storageKey = "streamlit_tab::" + cfg.page_key;

    const normalizeExpected = (txt) =>
        String(txt || "")
            .replace(/:material\\/[^:]+:/g, "")
            .replace(/\\s+/g, " ")
            .trim()
            .toLowerCase();

    const normalizeActual = (txt) =>
        String(txt || "")
            .replace(/\\s+/g, " ")
            .trim()
            .toLowerCase();

    function findMatchingTabs() {{
        const parentDoc = window.parent && window.parent.document;
        if (!parentDoc) return null;

        const expected = cfg.labels.map(normalizeExpected);
        const tabLists = Array.from(parentDoc.querySelectorAll('div[role="tablist"]'));

        for (const tabList of tabLists) {{
            const buttons = Array.from(tabList.querySelectorAll('button[role="tab"]'));
            if (buttons.length !== expected.length) continue;

            const actual = buttons.map((btn) => normalizeActual(btn.innerText));
            const matches = expected.every((label, idx) => label && actual[idx].includes(label));
            if (matches) return buttons;
        }}

        return null;
    }}

    function bindAndRestore() {{
        const buttons = findMatchingTabs();
        if (!buttons) return false;

        buttons.forEach((btn, idx) => {{
            if (btn.dataset.persistTabBound === "1") return;
            btn.dataset.persistTabBound = "1";
            btn.addEventListener("click", () => {{
                try {{
                    window.localStorage.setItem(storageKey, String(idx));
                }} catch (e) {{}}
            }});
        }});

        try {{
            const raw = window.localStorage.getItem(storageKey);
            if (raw === null) return true;

            const savedIndex = Number.parseInt(raw, 10);
            if (Number.isNaN(savedIndex) || savedIndex < 0 || savedIndex >= buttons.length) return true;

            const activeIndex = buttons.findIndex((btn) => btn.getAttribute("aria-selected") === "true");
            if (activeIndex !== savedIndex) {{
                buttons[savedIndex].click();
            }}
        }} catch (e) {{}}

        return true;
    }}

    let attempts = 0;
    const timer = setInterval(() => {{
        attempts += 1;
        if (bindAndRestore() || attempts > 50) {{
            clearInterval(timer);
        }}
    }}, 100);
}})();
</script>
    """,
        height="content",
    width="stretch",
    )
