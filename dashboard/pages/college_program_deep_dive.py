import plotly.express as px
import streamlit as st

from dashboard.constants import COLOR_MAP, STYPE_MAP
from dashboard.ui import dark_layout, kpi, safe_course_name


def render(df_full):
    st.markdown("# 🎓 College / Program Deep Dive")

    sel_cp_uni = st.selectbox("University", sorted(df_full["University"].dropna().unique()))
    cp_uni_df = df_full[df_full["University"] == sel_cp_uni]

    sel_cp_col = st.selectbox("College", sorted(cp_uni_df["College"].dropna().unique()))
    cp_df = cp_uni_df[cp_uni_df["College"] == sel_cp_col].copy()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, len(cp_df), "Students in College", "")
    kpi(c2, f"{(cp_df['Predicted_Target']=='Dropout').mean()*100:.1f}%", "College Dropout Rate", "", "#f87171")
    kpi(c3, f"{(cp_df['Predicted_Target']=='Graduate').mean()*100:.1f}%", "Graduation Rate", "", "#34d399")
    kpi(c4, cp_df["Program"].nunique(), "Programs", "")
    st.markdown("<br>", unsafe_allow_html=True)

    tab_prog, tab_risk_heat, tab_grade, tab_stype = st.tabs(
        ["📊 By Program", "🔥 Risk Heatmap", "📐 Grade Analysis", "👥 Student Types"]
    )

    with tab_prog:
        prog_stats = (
            cp_df.groupby("Program")
            .agg(
                Students=("Risk_Score", "count"),
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                Grad_Rate=("Predicted_Target", lambda x: (x == "Graduate").mean()),
                High_Risk=("Risk_Label", lambda x: (x == "High").mean()),
                Avg_Grade=("Curricular units 2nd sem (grade)", "mean"),
            )
            .reset_index()
            .sort_values("Dropout_Rate", ascending=False)
        )

        r1, r2 = st.columns(2)

        fig_prog_d = px.bar(
            prog_stats,
            x="Program",
            y="Dropout_Rate",
            color="Dropout_Rate",
            color_continuous_scale="Reds",
            title="Dropout Rate by Program",
            text_auto=".1%",
        )
        dark_layout(fig_prog_d, height=360)
        r1.plotly_chart(fig_prog_d, use_container_width=True)

        fig_prog_g2 = px.bar(
            prog_stats,
            x="Program",
            y="Grad_Rate",
            color="Grad_Rate",
            color_continuous_scale="Greens",
            title="Graduation Rate by Program",
            text_auto=".1%",
        )
        dark_layout(fig_prog_g2, height=360)
        r2.plotly_chart(fig_prog_g2, use_container_width=True)

        top_prog_grade = (
            cp_df.groupby("Program")
            .agg(Avg_Grade=("Curricular units 2nd sem (grade)", "mean"), Students=("Risk_Score", "count"))
            .reset_index()
            .sort_values("Avg_Grade", ascending=False)
            .head(10)
        )
        fig_prog_grade = px.bar(
            top_prog_grade,
            x="Program",
            y="Avg_Grade",
            color="Avg_Grade",
            color_continuous_scale="Blues",
            title="Avg 2nd Sem Grade by Program (Top 10)",
            labels={"Avg_Grade": "Avg Grade"},
        )
        dark_layout(fig_prog_grade, height=380)
        fig_prog_grade.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_prog_grade, use_container_width=True)

        if "Course" in cp_df.columns:
            cp_df_plot = cp_df.copy()
            cp_df_plot["Course_Name"] = cp_df_plot["Course"].apply(safe_course_name)
            course_risk = (
                cp_df_plot.groupby("Course_Name")
                .agg(
                    Students=("Risk_Score", "count"),
                    Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                )
                .reset_index()
                .sort_values("Dropout_Rate", ascending=False)
                .head(12)
            )
            fig_course_risk = px.bar(
                course_risk,
                y="Course_Name",
                x="Dropout_Rate",
                orientation="h",
                color="Dropout_Rate",
                color_continuous_scale="Reds",
                title="Top 12 Courses by Predicted Dropout Rate",
                labels={"Dropout_Rate": "Dropout Rate", "Course_Name": "Course"},
            )
            dark_layout(fig_course_risk, height=420)
            fig_course_risk.update_layout(yaxis_title="")
            st.plotly_chart(fig_course_risk, use_container_width=True)

        st.dataframe(
            prog_stats.style.format(
                {
                    "Dropout_Rate": "{:.1%}",
                    "Grad_Rate": "{:.1%}",
                    "High_Risk": "{:.1%}",
                    "Avg_Grade": "{:.2f}",
                }
            ).background_gradient(subset=["Dropout_Rate"], cmap="Reds"),
            use_container_width=True,
        )

    with tab_risk_heat:
        heat = cp_df.groupby(["Program", "Risk_Label"]).size().unstack(fill_value=0)
        heat = heat.reindex(columns=["High", "Medium", "Low"], fill_value=0)
        fig_heat = px.imshow(
            heat,
            color_continuous_scale="RdYlGn_r",
            title=f"Risk Level Heatmap – {sel_cp_col}",
            labels={"color": "Count"},
            text_auto=True,
        )
        dark_layout(fig_heat, height=380)
        st.plotly_chart(fig_heat, use_container_width=True)

        drop_cp = cp_df[(cp_df["Dropout_Reason"] != "") & (cp_df["Dropout_Reason"].notna())]
        if not drop_cp.empty:
            heat2 = drop_cp.groupby(["Program", "Dropout_Reason"]).size().unstack(fill_value=0)
            fig_heat2 = px.imshow(
                heat2,
                color_continuous_scale="Reds",
                title=f"Dropout Reasons × Program – {sel_cp_col}",
                labels={"color": "Count"},
                text_auto=True,
            )
            dark_layout(fig_heat2, height=380)
            st.plotly_chart(fig_heat2, use_container_width=True)

    with tab_grade:
        r1, r2 = st.columns(2)

        fig_g_prog = px.box(
            cp_df,
            x="Program",
            y="Curricular units 2nd sem (grade)",
            color="Program",
            title="2nd Semester Grade by Program",
        )
        dark_layout(fig_g_prog, height=380)
        r1.plotly_chart(fig_g_prog, use_container_width=True)

        fig_g_out = px.violin(
            cp_df,
            x="Predicted_Target_Display",
            y="Curricular units 2nd sem (grade)",
            color="Predicted_Target_Display",
            color_discrete_map=COLOR_MAP,
            box=True,
            title="Grade by Predicted Outcome",
        )
        dark_layout(fig_g_out, height=380)
        r2.plotly_chart(fig_g_out, use_container_width=True)

        fig_scatter_cp = px.scatter(
            cp_df.sample(min(600, len(cp_df)), random_state=42),
            x="Curricular units 1st sem (grade)",
            y="Curricular units 2nd sem (grade)",
            color="Predicted_Target_Display",
            color_discrete_map=COLOR_MAP,
            symbol="Program",
            opacity=0.65,
            title="1st vs 2nd Sem Grades by Program & Outcome",
        )
        dark_layout(fig_scatter_cp, height=420)
        st.plotly_chart(fig_scatter_cp, use_container_width=True)

    with tab_stype:
        r1, r2 = st.columns(2)

        stype_prog = cp_df.groupby(["Program", "Student_Type"]).size().reset_index(name="Count")
        fig_stype_p = px.bar(
            stype_prog,
            x="Program",
            y="Count",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            barmode="stack",
            title="Student Type by Program",
        )
        dark_layout(fig_stype_p, height=360)
        r1.plotly_chart(fig_stype_p, use_container_width=True)

        stype_drop = cp_df.groupby(["Student_Type", "Predicted_Target_Display"]).size().reset_index(name="Count")
        fig_stype_out = px.bar(
            stype_drop,
            x="Student_Type",
            y="Count",
            color="Predicted_Target_Display",
            color_discrete_map=COLOR_MAP,
            barmode="stack",
            title="Outcomes by Student Type",
        )
        dark_layout(fig_stype_out, height=360)
        r2.plotly_chart(fig_stype_out, use_container_width=True)
