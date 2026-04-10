"""
app.py - Student Performance & Predictive Analytics Dashboard (UAE Edition)
Run:  streamlit run app.py
"""

import os

import streamlit as st

from dashboard.constants import OUTCOME_DISPLAY_ORDER, PAGES, display_outcome
from dashboard.data import (
    DATA_CSV,
    add_display_columns,
    filter_dataframe,
    get_trainer,
    load_scores,
    merge_enrichment_columns,
    need_training,
    run_training,
)
from dashboard.pages import (
    analytics,
    at_risk,
    college_program_deep_dive,
    emirati_vs_expats,
    macro_economic,
    model_performance,
    overview,
    predict_new_student,
    student_deep_dive,
    students_abroad,
    university_comparison,
    university_deep_dive,
)
from dashboard.styles import inject_global_styles

st.set_page_config(
    page_title="UAE Student Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()


with st.sidebar:
    st.markdown("## 🎓 UAE Student Analytics")
    st.markdown("---")

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")

    data_ok = os.path.exists(DATA_CSV)
    model_ok = not need_training()

    st.markdown("### Status")
    st.markdown(f"{'✅' if data_ok else '❌'} dataset `data.csv`")
    st.markdown(f"{'✅' if model_ok else '⏳'} trained model")

    if not data_ok:
        st.warning("Place `data.csv` in the same folder, then refresh.")
        st.stop()

    if not model_ok:
        if st.button("▶ Train Model Now", type="primary", use_container_width=True):
            run_training()
            st.rerun()
        st.info("Click the button above to train the model on first run.")
        st.stop()

    st.markdown("---")

    st.markdown("### 🔎 Global Filters")
    df_full = load_scores()
    if df_full is None:
        st.error("Scores not found. Re-train the model.")
        st.stop()

    df_full = merge_enrichment_columns(df_full)

    all_unis = sorted(df_full["University"].dropna().unique().tolist())
    sel_unis = st.multiselect("University", all_unis, default=all_unis)

    avail_colleges = sorted(
        df_full[df_full["University"].isin(sel_unis)]["College"].dropna().unique().tolist()
    )
    sel_colleges = st.multiselect("College", avail_colleges, default=avail_colleges)

    avail_programs = sorted(
        df_full[
            df_full["University"].isin(sel_unis) & df_full["College"].isin(sel_colleges)
        ]["Program"]
        .dropna()
        .unique()
        .tolist()
    )
    sel_programs = st.multiselect("Program", avail_programs, default=avail_programs)

    all_stypes = ["Emirati", "Expat", "Abroad"]
    sel_stypes = st.multiselect("Student Type", all_stypes, default=all_stypes)

    risk_filter = st.multiselect(
        "Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )

    st.markdown("---")
    st.caption("Model: Stacking ensemble\nXGB + LGBM + CatBoost -> XGB meta")
    st.caption(
        "Outcome note: Pending means the model does not yet have enough evidence for a confident final outcome."
    )


df = filter_dataframe(
    df_full, sel_unis, sel_colleges, sel_programs, sel_stypes, risk_filter
)
df_full, df = add_display_columns(df_full, df, display_outcome)


ROUTES = {
    "📊 Overview": lambda: overview.render(df),
    "🚨 At-Risk Students": lambda: at_risk.render(df, display_outcome),
    "📈 Analytics": analytics.render,
    "🌍 Macro-Economic": lambda: macro_economic.render(df),
    "🔍 Student Deep-Dive": lambda: student_deep_dive.render(
        df, df_full, display_outcome, OUTCOME_DISPLAY_ORDER
    ),
    "🤖 Predict New Student": lambda: predict_new_student.render(
        display_outcome, OUTCOME_DISPLAY_ORDER
    ),
    "📉 Model Performance": lambda: model_performance.render(
        df_full, get_trainer, display_outcome
    ),
    "🏛️ University Deep Dive": lambda: university_deep_dive.render(df_full),
    "🇦🇪 Emirati vs Expats": lambda: emirati_vs_expats.render(df_full, display_outcome),
    "✈️ Students Abroad": lambda: students_abroad.render(df_full, display_outcome),
    "🎓 College / Program Deep Dive": lambda: college_program_deep_dive.render(df_full),
    "🏆 University Comparison": lambda: university_comparison.render(df_full),
}

if page not in ROUTES:
    st.error(f"Unknown page: {page}")
    st.stop()

ROUTES[page]()
