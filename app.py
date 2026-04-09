"""
app.py – Student Performance & Predictive Analytics Dashboard
Run:  streamlit run app.py
"""

import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Student Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a3040; }
h1,h2,h3,h4 { color: #e8eaf0; }
p, li, label { color: #b0b8c8; }

/* ── KPI cards ── */
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

/* ── Risk badges ── */
.badge-high   { background:#3d1212; color:#f87171; border:1px solid #f87171;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}
.badge-medium { background:#3d2d0a; color:#fbbf24; border:1px solid #fbbf24;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}
.badge-low    { background:#0d2b1a; color:#34d399; border:1px solid #34d399;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600;}

/* ── section headers ── */
.section-header {
    font-size:1.15rem; font-weight:700; color:#93c5fd;
    border-left:3px solid #3b82f6; padding-left:10px; margin:24px 0 12px;
}

/* ── intervention card ── */
.intervention-card {
    background: #131c2e;
    border: 1px solid #2a3f6f;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.intervention-title { font-weight:700; color:#93c5fd; }

/* ── dataframe tweaks ── */
[data-testid="stDataFrame"] { border-radius: 8px; }

/* ── progress bar ── */
.stProgress > div > div { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ── Lazy imports (heavy; loaded once) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_trainer():
    import model_trainer
    return model_trainer


# ── Data / model loading ──────────────────────────────────────────────────────
DATA_CSV   = "data.csv"
SCORES_CSV = "models/all_student_scores.csv"

def _need_training():
    return not (
        os.path.exists("models/stacking_ensemble.joblib") and
        os.path.exists(SCORES_CSV)
    )


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    mt = get_trainer()
    df = mt.load_scores()
    return df


def run_training():
    mt = get_trainer()
    with st.spinner("🔄 Training ensemble model on all students … (first run only – ~2-5 min)"):
        df = mt.train_and_save(DATA_CSV)
    st.cache_data.clear()
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Analytics")
    st.markdown("---")

    # UPDATED: Added Model Performance page
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🚨 At-Risk Students", "📈 Analytics", "🔍 Student Deep-Dive", "🤖 Predict New Student", "📉 Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Data source status ────────────────────────────────────────────────────
    data_ok   = os.path.exists(DATA_CSV)
    model_ok  = not _need_training()

    st.markdown("### Status")
    st.markdown(f"{'✅' if data_ok  else '❌'} dataset `data.csv`")
    st.markdown(f"{'✅' if model_ok else '⏳'} trained model")

    if not data_ok:
        st.warning(
            "Place `data.csv` (from the UCI / Kaggle dataset) in the same folder, then refresh."
        )
        st.stop()

    if not model_ok:
        if st.button("▶ Train Model Now", type="primary", use_container_width=True):
            run_training()
            st.rerun()
        else:
            st.info("Click the button above to train the model on first run.")
            st.stop()

    st.markdown("---")

    # ── Global filters (shared across pages) ──────────────────────────────────
    st.markdown("### Filters")
    df_full = load_scores()
    if df_full is None:
        st.error("Scores not found. Re-train the model.")
        st.stop()

    risk_filter = st.multiselect(
        "Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
    
    # REMOVED: Actual Outcome filter to simulate 'future unknown target'

    st.markdown("---")
    st.caption("Model: Stacking ensemble  \nXGB + LGBM + CatBoost → XGB meta")


# ── Apply global filters ──────────────────────────────────────────────────────
# Only filter by Risk Label now
df = df_full[df_full["Risk_Label"].isin(risk_filter)].copy()
df.index = range(len(df))

# ── Helper utilities ──────────────────────────────────────────────────────────
COLOR_MAP = {"Dropout": "#f87171", "Enrolled": "#fbbf24", "Graduate": "#34d399"}
RISK_MAP  = {"High": "#f87171",    "Medium": "#fbbf24",   "Low": "#34d399"}

COURSE_MAP = {
    33:"Biofuel Production Technologies", 171:"Animation & Multimedia Design",
    8014:"Social Service (evening)", 9003:"Agronomy", 9070:"Communication Design",
    9085:"Veterinary Nursing", 9119:"Informatics Engineering", 9130:"Equinculture",
    9147:"Management", 9238:"Social Service", 9254:"Tourism", 9500:"Nursing",
    9556:"Oral Hygiene", 9670:"Advertising & Marketing Management",
    9773:"Journalism & Communication", 9853:"Basic Education", 9991:"Management (evening)",
}

def safe_course_name(code):
    try:
        return COURSE_MAP.get(int(code), f"Course {code}")
    except Exception:
        return str(code)


def risk_badge(r):
    cls = f"badge-{str(r).lower()}"
    return f'<span class="{cls}">{r}</span>'


def gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
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
                {"range": [0, 30],  "color": "#0d2b1a"},
                {"range": [30, 60], "color": "#3d2d0a"},
                {"range": [60, 100],"color": "#3d1212"},
            ],
        },
    ))
    fig.update_layout(
        height=200, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f1117", font_color="#b0b8c8",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Student Performance Overview")
    st.markdown("Real-time risk analytics powered by a stacking ensemble model.")

    total   = len(df_full)
    at_risk = (df_full["Risk_Label"] == "High").sum()
    
    # UPDATED: Use Predicted_Target instead of actual Target
    dropout_rate  = (df_full["Predicted_Target"] == "Dropout").mean()
    grad_rate     = (df_full["Predicted_Target"] == "Graduate").mean()
    enrolled_rate = (df_full["Predicted_Target"] == "Enrolled").mean()
    
    low_engage    = (df_full["Engagement_Flag"] == "Low Engagement").sum()

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, delta in [
        (c1, total,          "Total Students",         ""),
        (c2, at_risk,        "High-Risk Students",     f"{at_risk/total*100:.1f}% of total"),
        (c3, f"{dropout_rate*100:.1f}%", "Predicted Dropout Rate", ""),
        (c4, f"{grad_rate*100:.1f}%",   "Predicted Graduation Rate", ""),
        (c5, low_engage,     "Low Engagement",         "Enrolled & 0 approved units"),
    ]:
        col.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value">{val}</p>'
            f'<p class="kpi-label">{lbl}</p>'
            f'<p class="kpi-delta" style="color:#8090a8">{delta}</p>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauges row ────────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge_chart(dropout_rate,  "Predicted Dropout Rate",     "#f87171"), use_container_width=True)
    g2.plotly_chart(gauge_chart(enrolled_rate, "Predicted Still Enrolled",   "#fbbf24"), use_container_width=True)
    g3.plotly_chart(gauge_chart(grad_rate,     "Predicted Graduation Rate",  "#34d399"), use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    row1_l, row1_r = st.columns(2)

    # UPDATED: Predicted Target distribution
    dist = df_full["Predicted_Target"].value_counts().reset_index()
    dist.columns = ["Predicted_Target", "Count"]
    fig_pie = px.pie(
        dist, names="Predicted_Target", values="Count",
        color="Predicted_Target", color_discrete_map=COLOR_MAP,
        hole=0.55, title="Predicted Outcome Distribution"
    )
    fig_pie.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                          font_color="#b0b8c8", legend_font_color="#b0b8c8")
    row1_l.plotly_chart(fig_pie, use_container_width=True)

    # Risk distribution
    risk_dist = df_full["Risk_Label"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
    risk_dist.columns = ["Risk", "Count"]
    fig_risk = px.bar(
        risk_dist, x="Risk", y="Count", color="Risk",
        color_discrete_map=RISK_MAP,
        title="Students by Risk Level", text="Count"
    )
    fig_risk.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                           font_color="#b0b8c8", showlegend=False)
    fig_risk.update_traces(textposition="outside")
    row1_r.plotly_chart(fig_risk, use_container_width=True)

    # Dropout probability histogram
    row2_l, row2_r = st.columns(2)
    fig_hist = px.histogram(
        df_full, x="Risk_Score", color="Predicted_Target",
        color_discrete_map=COLOR_MAP, nbins=40,
        title="Dropout Probability Distribution",
        labels={"Risk_Score": "P(Dropout)", "count": "Students"},
        barmode="overlay", opacity=0.75,
    )
    fig_hist.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                           font_color="#b0b8c8")
    row2_l.plotly_chart(fig_hist, use_container_width=True)

    # UPDATED: Course-level dropout rates based on Prediction
    df_full["Course_Name"] = df_full["Course"].apply(safe_course_name)
    course_risk = (
        df_full.groupby("Course_Name")
               .apply(lambda g: (g["Predicted_Target"] == "Dropout").mean())
               .reset_index()
               .rename(columns={0: "Predicted Dropout Rate"})
               .sort_values("Predicted Dropout Rate", ascending=False)
               .head(12)
    )
    fig_course = px.bar(
        course_risk, y="Course_Name", x="Predicted Dropout Rate", orientation="h",
        title="Top 12 Courses by Predicted Dropout Rate",
        color="Predicted Dropout Rate", color_continuous_scale="Reds",
    )
    fig_course.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                             font_color="#b0b8c8", yaxis_title="")
    row2_r.plotly_chart(fig_course, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – AT-RISK STUDENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 At-Risk Students":
    st.markdown("# 🚨 Early Warning System")

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    tab_dropout, tab_engage = st.tabs(["🔴 Dropout Risk", "🟡 Low Engagement"])

    # ── Dropout Risk ─────────────────────────────────────────────────────────
    with tab_dropout:
        high_risk = df[df["Risk_Label"] == "High"].sort_values("Risk_Score", ascending=False)
        st.markdown(f"<p class='section-header'>Showing {len(high_risk)} high-risk students (filtered)</p>",
                    unsafe_allow_html=True)

        # Summary strip
        s1, s2, s3 = st.columns(3)
        s1.metric("High-Risk Students", len(high_risk))
        s2.metric("Avg Dropout Prob",   f"{high_risk['Risk_Score'].mean()*100:.1f}%")
        # REMOVED: "Already Dropped Out" metric

        st.markdown("---")

        # Table
        display_cols = [
            "Predicted_Target", "Risk_Score", "Risk_Label",
            "Gender", "Age at enrollment", "Scholarship holder", "Debtor",
            "Tuition fees up to date",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
            "Prob_Dropout", "Prob_Graduate",
        ]
        display_cols = [c for c in display_cols if c in high_risk.columns]

        st.dataframe(
            high_risk[display_cols]
                .rename(columns={
                    "Predicted_Target": "Predicted",
                    "Risk_Score": "P(Dropout)", "Risk_Label": "Risk",
                    "Age at enrollment": "Age", "Scholarship holder": "Scholar",
                    "Tuition fees up to date": "Fees OK",
                    "Curricular units 2nd sem (approved)": "Units Approved (S2)",
                    "Curricular units 2nd sem (grade)": "Grade S2",
                })
                .style
                .background_gradient(subset=["P(Dropout)"], cmap="Reds")
                .format({"P(Dropout)": "{:.2%}", "Prob_Graduate": "{:.2%}", "Grade S2": "{:.2f}"}),
            use_container_width=True, height=500,
        )

        # Download
        st.download_button(
            "⬇ Download High-Risk List (CSV)",
            high_risk[display_cols].to_csv(index=False).encode(),
            "high_risk_students.csv", "text/csv",
        )

        # # Visual
        # # UPDATED: Color by Predicted_Target
        # fig_scatter = px.scatter(
        #     high_risk, x="Curricular units 2nd sem (approved)",
        #     y="Risk_Score", color="Predicted_Target",
        #     color_discrete_map=COLOR_MAP, size="Prob_Dropout",
        #     hover_data=["Age at enrollment", "Scholarship holder", "Debtor"],
        #     title="Dropout Risk vs Units Approved (2nd Sem)",
        #     labels={"Risk_Score": "P(Dropout)", "Curricular units 2nd sem (approved)": "Units Approved"},
        # )
        # fig_scatter.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        #                            font_color="#b0b8c8")
        # st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Low Engagement ────────────────────────────────────────────────────────
    with tab_engage:
        low_eng = df[df["Engagement_Flag"] == "Low Engagement"].sort_values("Risk_Score", ascending=False)
        st.markdown(f"<p class='section-header'>Low Engagement Students: {len(low_eng)}</p>",
                    unsafe_allow_html=True)
        st.caption("Enrolled students with 0 approved curricular units in 2nd semester — potential silent dropout risk.")

        e1, e2, e3 = st.columns(3)
        e1.metric("Low Engagement", len(low_eng))
        e2.metric("Avg Dropout Prob", f"{low_eng['Risk_Score'].mean()*100:.1f}%")
        e3.metric("High-Risk within group", (low_eng["Risk_Label"] == "High").sum())

        engage_cols = [
            "Predicted_Target", "Risk_Score",
            "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (grade)",
            "Scholarship holder", "Debtor", "Tuition fees up to date",
        ]
        engage_cols = [c for c in engage_cols if c in low_eng.columns]
        st.dataframe(low_eng[engage_cols].style.background_gradient(subset=["Risk_Score"], cmap="Oranges"),
                     use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.markdown("# 📈 Analytics & Insights")

    # ── Feature importance ────────────────────────────────────────────────────
    mt = get_trainer()
    art, le_obj, feat_cols, thresholds = mt.load_artefacts()
    lgbm_model = next(m for name, m in art["base_models"] if name == "lgbm")

    imp_df = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": lgbm_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(20)

    fig_imp = px.bar(
        imp_df, y="Feature", x="Importance", orientation="h",
        title="Top 20 Feature Importances (LightGBM)",
        color="Importance", color_continuous_scale="Blues",
    )
    fig_imp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                          font_color="#b0b8c8", yaxis_title="", height=600)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # ── Grade distributions ───────────────────────────────────────────────────
    row_l, row_r = st.columns(2)

    # UPDATED: Use Predicted_Target
    fig_grade = px.box(
        df_full, x="Predicted_Target", y="Curricular units 2nd sem (grade)",
        color="Predicted_Target", color_discrete_map=COLOR_MAP,
        title="2nd Semester Grade by Predicted Outcome",
        points="outliers",
    )
    fig_grade.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                            font_color="#b0b8c8", showlegend=False)
    row_l.plotly_chart(fig_grade, use_container_width=True)

    fig_units = px.box(
        df_full, x="Predicted_Target", y="Curricular units 2nd sem (approved)",
        color="Predicted_Target", color_discrete_map=COLOR_MAP,
        title="2nd Semester Units Approved by Predicted Outcome",
        points="outliers",
    )
    fig_units.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                            font_color="#b0b8c8", showlegend=False)
    row_r.plotly_chart(fig_units, use_container_width=True)

    # ── Age vs dropout ────────────────────────────────────────────────────────
    row2_l, row2_r = st.columns(2)

    # UPDATED: Use Predicted_Target
    fig_age = px.histogram(
        df_full, x="Age at enrollment", color="Predicted_Target",
        color_discrete_map=COLOR_MAP, barmode="overlay", opacity=0.7,
        nbins=30, title="Age at Enrollment by Predicted Outcome",
    )
    fig_age.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                          font_color="#b0b8c8")
    row2_l.plotly_chart(fig_age, use_container_width=True)

    # Scholarship vs dropout
    sch = df_full.groupby(["Scholarship holder", "Predicted_Target"]).size().reset_index(name="Count")
    sch["Scholarship holder"] = sch["Scholarship holder"].map({1: "Scholar", 0: "No Scholar"})
    fig_sch = px.bar(
        sch, x="Scholarship holder", y="Count", color="Predicted_Target",
        color_discrete_map=COLOR_MAP, barmode="group",
        title="Scholarship vs Predicted Outcome",
    )
    fig_sch.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                          font_color="#b0b8c8")
    row2_r.plotly_chart(fig_sch, use_container_width=True)

    # ── Macro-economics ───────────────────────────────────────────────────────
    # st.markdown("<p class='section-header'>Macro-economic Factors</p>", unsafe_allow_html=True)
    # r3l, r3r = st.columns(2)

    # # FIX: Removed trendline="lowess" to avoid statsmodels ModuleNotFoundError
    # # UPDATED: Color by Predicted_Target
    # fig_unemp = px.scatter(
    #     df_full, x="Unemployment rate", y="Risk_Score",
    #     color="Predicted_Target", color_discrete_map=COLOR_MAP, opacity=0.4,
    #     title="Unemployment Rate vs Dropout Risk"
    # )
    # fig_unemp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    #                          font_color="#b0b8c8")
    # r3l.plotly_chart(fig_unemp, use_container_width=True)

    # fig_gdp = px.scatter(
    #     df_full, x="GDP", y="Risk_Score",
    #     color="Predicted_Target", color_discrete_map=COLOR_MAP, opacity=0.4,
    #     title="GDP vs Dropout Risk"
    # )
    # fig_gdp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    #                        font_color="#b0b8c8")
    # r3r.plotly_chart(fig_gdp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – STUDENT DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Student Deep-Dive":
    st.markdown("# 🔍 Individual Student Analysis")

    col_sel, col_sort = st.columns([3, 1])
    with col_sort:
        sort_by = st.selectbox("Sort list by", ["Risk Score ↓", "Index ↑"], index=0)

    sorted_df = df_full.sort_values(
        "Risk_Score", ascending=(sort_by == "Index ↑")
    ).reset_index(drop=True)

    # UPDATED: Remove "Actual" from selectbox text
    student_idx = col_sel.selectbox(
        "Select student (row index in full dataset)",
        sorted_df.index,
        format_func=lambda i: (
            f"Student #{i} | Risk: {sorted_df.loc[i,'Risk_Label']} "
            f"({sorted_df.loc[i,'Risk_Score']*100:.1f}%)"
        )
    )

    stu = sorted_df.loc[student_idx]

    # ── Header ────────────────────────────────────────────────────────────────
    # REMOVED: Actual Outcome Card
    h1, h2, h3 = st.columns(3)
    h1.markdown(
        f'<div class="kpi-card"><p class="kpi-value">'
        f'{stu["Risk_Score"]*100:.1f}%</p><p class="kpi-label">Dropout Risk</p></div>',
        unsafe_allow_html=True
    )
    h2.markdown(
        f'<div class="kpi-card"><p class="kpi-value">{stu["Predicted_Target"]}</p>'
        f'<p class="kpi-label">Model Prediction</p></div>', unsafe_allow_html=True
    )
    risk_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}[str(stu["Risk_Label"])]
    h3.markdown(
        f'<div class="kpi-card"><p class="kpi-value" style="color:{risk_color}">'
        f'{stu["Risk_Label"]}</p><p class="kpi-label">Risk Level</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability bar ───────────────────────────────────────────────────────
    fig_prob = go.Figure(go.Bar(
        x=[stu["Prob_Dropout"], stu["Prob_Enrolled"], stu["Prob_Graduate"]],
        y=["Dropout", "Enrolled", "Graduate"], orientation="h",
        marker_color=["#f87171", "#fbbf24", "#34d399"],
        text=[f"{v*100:.1f}%" for v in [stu["Prob_Dropout"], stu["Prob_Enrolled"], stu["Prob_Graduate"]]],
        textposition="outside",
    ))
    fig_prob.update_layout(
        title="Prediction Probabilities", paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117", font_color="#b0b8c8",
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=200, margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Profile grid ──────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Student Profile</p>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**Demographics**")
        st.markdown(f"- Gender: {'Male' if stu.get('Gender')==1 else 'Female'}")
        st.markdown(f"- Age at enrollment: {stu.get('Age at enrollment', 'N/A')}")
        st.markdown(f"- International: {'Yes' if stu.get('International')==1 else 'No'}")
        st.markdown(f"- Course: {safe_course_name(stu.get('Course', 'N/A'))}")

    with p2:
        st.markdown("**Financial**")
        st.markdown(f"- Scholarship: {'✅' if stu.get('Scholarship holder')==1 else '❌'}")
        st.markdown(f"- Debtor: {'⚠️ Yes' if stu.get('Debtor')==1 else 'No'}")
        st.markdown(f"- Fees up to date: {'✅' if stu.get('Tuition fees up to date')==1 else '❌'}")
        st.markdown(f"- Displaced: {'Yes' if stu.get('Displaced')==1 else 'No'}")

    with p3:
        st.markdown("**Academic (2nd Semester)**")
        st.markdown(f"- Units enrolled: {stu.get('Curricular units 2nd sem (enrolled)', 'N/A')}")
        st.markdown(f"- Units approved: {stu.get('Curricular units 2nd sem (approved)', 'N/A')}")
        st.markdown(f"- Grade: {stu.get('Curricular units 2nd sem (grade)', 0):.2f}")
        st.markdown(f"- Evaluations: {stu.get('Curricular units 2nd sem (evaluations)', 'N/A')}")

    # ── Radar chart ───────────────────────────────────────────────────────────
    radar_features = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Admission grade",
        "Previous qualification (grade)",
    ]
    radar_features = [f for f in radar_features if f in df_full.columns]

    stu_norm = []
    avg_norm = []
    for feat in radar_features:
        col_min, col_max = df_full[feat].min(), df_full[feat].max()
        rng = col_max - col_min or 1
        stu_norm.append((stu[feat] - col_min) / rng)
        avg_norm.append((df_full[feat].mean() - col_min) / rng)

    labels_short = [f.replace("Curricular units ", "").replace(" sem ", "S").replace("(", "").replace(")", "")
                    for f in radar_features]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=stu_norm + [stu_norm[0]], theta=labels_short + [labels_short[0]],
                                        fill="toself", name="Student", line_color="#60a5fa"))
    fig_radar.add_trace(go.Scatterpolar(r=avg_norm + [avg_norm[0]], theta=labels_short + [labels_short[0]],
                                        fill="toself", name="Average", line_color="#a3e635", opacity=0.5))
    fig_radar.update_layout(
        polar=dict(bgcolor="#1a2035", radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor="#0f1117", font_color="#b0b8c8", title="Student vs Cohort Average",
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── AI-Generated Interventions ────────────────────────────────────────────
    st.markdown("<p class='section-header'>🤖 AI-Suggested Interventions</p>", unsafe_allow_html=True)

    def rule_interventions(s):
        tips = []
        if s.get("Debtor") == 1:
            tips.append(("💰 Financial Aid Referral",
                          "Student has outstanding debt. Connect with the financial aid office for payment plans or emergency grants.",
                          "High Priority"))
        if s.get("Tuition fees up to date") == 0:
            tips.append(("💳 Tuition Assistance",
                          "Tuition fees are not current. Risk of administrative withdrawal. Immediate intervention required.",
                          "High Priority"))
        if s.get("Curricular units 2nd sem (approved)", 5) == 0:
            tips.append(("📚 Academic Probation Review",
                          "Zero units approved in the 2nd semester signals serious academic difficulty. Schedule an urgent academic counselling session.",
                          "High Priority"))
        if s.get("Curricular units 2nd sem (grade)", 10) < 8 and s.get("Curricular units 2nd sem (grade)", 10) > 0:
            tips.append(("📖 Tutoring Programme",
                          f"2nd semester grade is {s.get('Curricular units 2nd sem (grade)', 0):.1f} — below passing threshold. Enroll in tutoring or peer-learning groups.",
                          "Medium Priority"))
        if s.get("Scholarship holder") == 0 and s.get("Debtor") == 1:
            tips.append(("🎓 Scholarship Application",
                          "Student is in debt without scholarship support. Advise on available merit/need-based scholarships.",
                          "Medium Priority"))
        if s.get("Age at enrollment", 20) > 30:
            tips.append(("🌐 Adult Learner Support",
                          "Mature student who may benefit from flexible scheduling, online resources, or mentoring from alumni with similar profiles.",
                          "Low Priority"))
        if s.get("International") == 1:
            tips.append(("🌍 International Student Office",
                          "International student — ensure visa status, housing, and language support are in order.",
                          "Low Priority"))
        if not tips:
            tips.append(("✅ On Track",
                          "No immediate red flags detected. Continue monitoring and provide encouragement.",
                          "Informational"))
        return tips

    pri_color = {"High Priority": "#f87171", "Medium Priority": "#fbbf24",
                 "Low Priority": "#34d399", "Informational": "#60a5fa"}

    for title, body, priority in rule_interventions(stu):
        clr = pri_color.get(priority, "#b0b8c8")
        st.markdown(
            f'<div class="intervention-card">'
            f'<span class="intervention-title">{title}</span>'
            f'<span style="float:right;font-size:0.75rem;color:{clr};font-weight:600">{priority}</span>'
            f'<br><span style="font-size:0.9rem;color:#b0b8c8">{body}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – PREDICT NEW STUDENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict New Student":
    st.markdown("# 🤖 Predict New Student Risk")
    st.markdown("Enter a student's information to get an instant risk prediction.")

    mt  = get_trainer()
    art, le_obj, feat_cols, thresholds = mt.load_artefacts()

    if art is None:
        st.error("Model not loaded.")
        st.stop()

    with st.form("predict_form"):
        st.markdown("<p class='section-header'>Demographics</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gender         = c1.selectbox("Gender",        ["Male", "Female"])
        age            = c2.number_input("Age at Enrollment", 17, 70, 22)
        international  = c3.selectbox("International", ["No", "Yes"])
        marital        = c1.selectbox("Marital Status",
                                      [1, 2, 3, 4, 5, 6],
                                      format_func=lambda x: {1:"Single",2:"Married",3:"Widower",4:"Divorced",5:"Facto union",6:"Legally separated"}[x])
        displaced      = c2.selectbox("Displaced",     ["No", "Yes"])
        special_needs  = c3.selectbox("Special Needs", ["No", "Yes"])

        st.markdown("<p class='section-header'>Financial</p>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        scholarship    = f1.selectbox("Scholarship Holder", ["No", "Yes"])
        debtor         = f2.selectbox("Debtor",             ["No", "Yes"])
        fees_ok        = f3.selectbox("Tuition Fees Up to Date", ["Yes", "No"])

        st.markdown("<p class='section-header'>Academic</p>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        admission_grade   = a1.slider("Admission Grade",          0.0, 200.0, 130.0)
        prev_qual_grade   = a2.slider("Previous Qualification Grade", 0.0, 200.0, 130.0)

        a3, a4 = st.columns(2)
        s1_enrolled  = a3.number_input("1st Sem – Units Enrolled",  0, 30, 6)
        s1_approved  = a4.number_input("1st Sem – Units Approved",  0, 30, 5)
        s1_grade     = a3.number_input("1st Sem – Grade (0-20)",    0.0, 20.0, 12.0)
        s1_evals     = a4.number_input("1st Sem – Evaluations",     0, 50, 6)
        s1_no_eval   = a3.number_input("1st Sem – No Evaluations",  0, 30, 0)
        s1_credited  = a4.number_input("1st Sem – Units Credited",  0, 30, 0)

        a5, a6 = st.columns(2)
        s2_enrolled  = a5.number_input("2nd Sem – Units Enrolled",  0, 30, 6)
        s2_approved  = a6.number_input("2nd Sem – Units Approved",  0, 30, 5)
        s2_grade     = a5.number_input("2nd Sem – Grade (0-20)",    0.0, 20.0, 12.0)
        s2_evals     = a6.number_input("2nd Sem – Evaluations",     0, 50, 6)
        s2_no_eval   = a5.number_input("2nd Sem – No Evaluations",  0, 30, 0)
        s2_credited  = a6.number_input("2nd Sem – Units Credited",  0, 30, 0)

        st.markdown("<p class='section-header'>Macro-Economic</p>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        unemployment = e1.number_input("Unemployment Rate (%)", 0.0, 20.0, 10.8)
        inflation    = e2.number_input("Inflation Rate (%)",    -5.0, 10.0, 1.4)
        gdp          = e3.number_input("GDP",                   -5.0, 5.0, 1.74)

        submitted = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)

    if submitted:
        # Build feature row matching the training columns
        row_data = {c: 0 for c in feat_cols}
        row_data.update({
            "Marital status": marital,
            "Application mode": 1,
            "Application order": 1,
            "Course": 9147,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Previous qualification (grade)": prev_qual_grade,
            "Nationality": 1,
            "Mother's qualification": 1,
            "Father's qualification": 1,
            "Mother's occupation": 1,
            "Father's occupation": 1,
            "Admission grade": admission_grade,
            "Displaced": 1 if displaced == "Yes" else 0,
            "Educational special needs": 1 if special_needs == "Yes" else 0,
            "Debtor": 1 if debtor == "Yes" else 0,
            "Tuition fees up to date": 1 if fees_ok == "Yes" else 0,
            "Gender": 1 if gender == "Male" else 0,
            "Scholarship holder": 1 if scholarship == "Yes" else 0,
            "Age at enrollment": age,
            "International": 1 if international == "Yes" else 0,
            "Curricular units 1st sem (credited)": s1_credited,
            "Curricular units 1st sem (enrolled)": s1_enrolled,
            "Curricular units 1st sem (evaluations)": s1_evals,
            "Curricular units 1st sem (approved)": s1_approved,
            "Curricular units 1st sem (grade)": s1_grade,
            "Curricular units 1st sem (without evaluations)": s1_no_eval,
            "Curricular units 2nd sem (credited)": s2_credited,
            "Curricular units 2nd sem (enrolled)": s2_enrolled,
            "Curricular units 2nd sem (evaluations)": s2_evals,
            "Curricular units 2nd sem (approved)": s2_approved,
            "Curricular units 2nd sem (grade)": s2_grade,
            "Curricular units 2nd sem (without evaluations)": s2_no_eval,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
        })

        X_new = pd.DataFrame([row_data])[feat_cols]

        n_classes = 3
        base_models = art["base_models"]
        stacked = np.zeros((1, n_classes * len(base_models)))
        for m_idx, (name, model) in enumerate(base_models):
            probs = model.predict_proba(X_new)
            stacked[:, m_idx * n_classes : (m_idx + 1) * n_classes] = probs

        pred_encoded = art["meta_model"].predict(stacked)[0]
        pred_proba   = art["meta_model"].predict_proba(stacked)[0]
        pred_label   = le_obj.inverse_transform([pred_encoded])[0]

        class_list = list(le_obj.classes_)
        p_dropout  = pred_proba[class_list.index("Dropout")]
        p_enrolled = pred_proba[class_list.index("Enrolled")]
        p_grad     = pred_proba[class_list.index("Graduate")]

        q60 = thresholds["q60"]
        q75 = thresholds["q75"]
        risk_level = "High" if p_dropout > q75 else ("Medium" if p_dropout > q60 else "Low")
        risk_clr   = risk_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}[risk_level]

        st.markdown("---")
        st.markdown("## 🔮 Prediction Result")

        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'<div class="kpi-card"><p class="kpi-value" style="color:{risk_clr}">{p_dropout*100:.1f}%</p><p class="kpi-label">Dropout Risk</p></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="kpi-card"><p class="kpi-value">{pred_label}</p><p class="kpi-label">Predicted Outcome</p></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="kpi-card"><p class="kpi-value" style="color:{risk_clr}">{risk_level}</p><p class="kpi-label">Risk Level</p></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="kpi-card"><p class="kpi-value">{p_grad*100:.1f}%</p><p class="kpi-label">P(Graduate)</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_new = go.Figure(go.Bar(
            x=[p_dropout, p_enrolled, p_grad],
            y=["Dropout", "Enrolled", "Graduate"], orientation="h",
            marker_color=["#f87171", "#fbbf24", "#34d399"],
            text=[f"{v*100:.1f}%" for v in [p_dropout, p_enrolled, p_grad]],
            textposition="outside",
        ))
        fig_new.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                              font_color="#b0b8c8", xaxis=dict(range=[0, 1.1], tickformat=".0%"),
                              height=220, margin=dict(t=20, b=10))
        st.plotly_chart(fig_new, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – MODEL PERFORMANCE (NEW PAGE)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Model Performance":
    st.markdown("# 📉 Model Performance Analysis")
    st.markdown("Evaluation of the model on historical data (Actual vs Predicted).")

    # We use df_full here because this is the specific page allowed to see actuals
    y_true = df_full['Target']
    y_pred = df_full['Predicted_Target']

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    st.markdown("### Overall Accuracy")
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.markdown("---")
    
    st.markdown("### Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    with col2:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=["Dropout", "Enrolled", "Graduate"])
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Dropout", "Enrolled", "Graduate"],
            y=["Dropout", "Enrolled", "Graduate"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font_color="#b0b8c8")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.info("This page displays actual historical outcomes for model validation purposes. All other dashboard pages operate on 'future' predictions.")