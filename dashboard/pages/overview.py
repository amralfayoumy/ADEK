import plotly.express as px
import streamlit as st

from dashboard.constants import COLOR_MAP, RISK_MAP, STYPE_MAP
from dashboard.ui import dark_layout, gauge_chart, kpi


def render(df):
    st.markdown("# :material/insights: Student Performance Overview")
    st.markdown("Real-time risk analytics powered by a stacking ensemble model.")

    total = len(df)
    at_risk = (df["Risk_Label"] == "High").sum()
    dropout_rate = (df["Predicted_Target"] == "Dropout").mean()
    grad_rate = (df["Predicted_Target"] == "Graduate").mean()
    enrolled_rt = (df["Predicted_Target"] == "Enrolled").mean()
    low_engage = (df["Engagement_Flag"] == "Low Engagement").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, total, "Total Students (filtered)", "")
    kpi(c2, at_risk, "High-Risk Students", f"{at_risk/max(total,1)*100:.1f}% of total", "#f87171")
    kpi(c3, f"{dropout_rate*100:.1f}%", "Predicted Dropout Rate", "")
    kpi(c4, f"{grad_rate*100:.1f}%", "Predicted Graduation Rate", "", "#34d399")
    kpi(c5, low_engage, "Low Engagement", "Pending prediction & 0 approved units", "#fbbf24")

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge_chart(dropout_rate, "Predicted Dropout Rate", "#f87171"), use_container_width=True)
    g2.plotly_chart(gauge_chart(enrolled_rt, "Predicted Pending Outcome Rate", "#fbbf24"), use_container_width=True)
    g3.plotly_chart(gauge_chart(grad_rate, "Predicted Graduation Rate", "#34d399"), use_container_width=True)

    row1_l, row1_r = st.columns(2)

    dist = df["Predicted_Target_Display"].value_counts().reset_index()
    dist.columns = ["Predicted_Target_Display", "Count"]
    fig_pie = px.pie(
        dist,
        names="Predicted_Target_Display",
        values="Count",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        hole=0.55,
        title="Predicted Outcome Distribution",
    )
    dark_layout(fig_pie)
    row1_l.plotly_chart(fig_pie, use_container_width=True)

    risk_dist = df["Risk_Label"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
    risk_dist.columns = ["Risk", "Count"]
    fig_risk = px.bar(
        risk_dist,
        x="Risk",
        y="Count",
        color="Risk",
        color_discrete_map=RISK_MAP,
        title="Students by Risk Level",
        text="Count",
    )
    dark_layout(fig_risk)
    fig_risk.update_layout(showlegend=False)
    fig_risk.update_traces(textposition="outside")
    row1_r.plotly_chart(fig_risk, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    fig_hist = px.histogram(
        df,
        x="Risk_Score",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        nbins=40,
        title="Dropout Probability Distribution",
        labels={"Risk_Score": "P(Dropout)", "count": "Students"},
        barmode="overlay",
        opacity=0.75,
    )
    dark_layout(fig_hist)
    row2_l.plotly_chart(fig_hist, use_container_width=True)

    if "Student_Type" in df.columns and not df.empty:
        seg_risk = (
            df.groupby("Student_Type")
            .agg(
                Students=("Risk_Score", "count"),
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
            )
            .reset_index()
            .sort_values("Dropout_Rate", ascending=False)
        )
        fig_seg = px.bar(
            seg_risk,
            x="Student_Type",
            y="Dropout_Rate",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            text_auto=".1%",
            title="Dropout Rate by Student Type",
            labels={"Dropout_Rate": "Rate"},
        )
        dark_layout(fig_seg)
        row2_r.plotly_chart(fig_seg, use_container_width=True)
    else:
        row2_r.info("Student-type segmentation is unavailable in the current dataset.")

    st.markdown("<p class='section-header'>Grade Analysis</p>", unsafe_allow_html=True)
    ga1, ga2 = st.columns(2)

    fig_box = px.box(
        df,
        x="Predicted_Target_Display",
        y="Curricular units 2nd sem (grade)",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        title="2nd Sem Grade by Predicted Outcome",
    )
    dark_layout(fig_box)
    ga1.plotly_chart(fig_box, use_container_width=True)

    fig_scatter = px.scatter(
        df.sample(min(800, len(df)), random_state=42),
        x="Curricular units 1st sem (grade)",
        y="Curricular units 2nd sem (grade)",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        opacity=0.6,
        title="1st vs 2nd Semester Grades",
    )
    dark_layout(fig_scatter)
    ga2.plotly_chart(fig_scatter, use_container_width=True)
