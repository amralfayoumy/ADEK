import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.constants import UNI_COLORS
from dashboard.ui import dark_layout


def render(df_full):
    st.markdown("# 🏆 University Comparison")
    st.markdown("Side-by-side benchmarking of all UAE universities in the dataset.")

    compare_unis = st.multiselect(
        "Select universities to compare",
        sorted(df_full["University"].dropna().unique()),
        default=sorted(df_full["University"].dropna().unique()),
    )

    cmp_df = df_full[df_full["University"].isin(compare_unis)].copy()

    uni_agg = (
        cmp_df.groupby("University")
        .agg(
            Students=("Risk_Score", "count"),
            Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
            Grad_Rate=("Predicted_Target", lambda x: (x == "Graduate").mean()),
            Pending_Rate=("Predicted_Target", lambda x: (x == "Enrolled").mean()),
            High_Risk=("Risk_Label", lambda x: (x == "High").mean()),
            Avg_Risk=("Risk_Score", "mean"),
            Avg_Grade=("Curricular units 2nd sem (grade)", "mean"),
            Emirati_Pct=("Student_Type", lambda x: (x == "Emirati").mean()),
            Expat_Pct=("Student_Type", lambda x: (x == "Expat").mean()),
            Abroad_Pct=("Student_Type", lambda x: (x == "Abroad").mean()),
        )
        .reset_index()
    )

    tab_bar, tab_radar, tab_trend, tab_table, tab_scatter = st.tabs(
        ["📊 Bar Charts", "🕸️ Radar", "📅 Trends", "📋 Table", "🔵 Scatter"]
    )

    with tab_bar:
        fig_do_vs_gr = px.bar(
            uni_agg,
            x="University",
            y=["Dropout_Rate", "Grad_Rate"],
            barmode="group",
            title="Dropout vs Graduation Rate by University",
            color_discrete_map={"Dropout_Rate": "#f87171", "Grad_Rate": "#34d399"},
            labels={"value": "Rate", "variable": "Metric"},
        )
        dark_layout(fig_do_vs_gr, height=380)
        st.plotly_chart(fig_do_vs_gr, use_container_width=True)

        r1, r2 = st.columns(2)

        fig_do = px.bar(
            uni_agg.sort_values("Dropout_Rate", ascending=False),
            x="University",
            y="Dropout_Rate",
            color="University",
            text_auto=".1%",
            title="Dropout Rate by University",
            color_discrete_sequence=UNI_COLORS,
        )
        dark_layout(fig_do, height=380)
        fig_do.update_layout(showlegend=False)
        r1.plotly_chart(fig_do, use_container_width=True)

        fig_gr = px.bar(
            uni_agg.sort_values("Grad_Rate", ascending=False),
            x="University",
            y="Grad_Rate",
            color="University",
            text_auto=".1%",
            title="Graduation Rate by University",
            color_discrete_sequence=UNI_COLORS,
        )
        dark_layout(fig_gr, height=380)
        fig_gr.update_layout(showlegend=False)
        r2.plotly_chart(fig_gr, use_container_width=True)

        fig_hr = px.bar(
            uni_agg.sort_values("High_Risk", ascending=False),
            x="University",
            y="High_Risk",
            color="University",
            text_auto=".1%",
            title="High-Risk Student % by University",
            color_discrete_sequence=UNI_COLORS,
        )
        dark_layout(fig_hr, height=380)
        fig_hr.update_layout(showlegend=False)
        r1.plotly_chart(fig_hr, use_container_width=True)

        fig_ag = px.bar(
            uni_agg.sort_values("Avg_Grade", ascending=False),
            x="University",
            y="Avg_Grade",
            color="University",
            text_auto=".2f",
            title="Avg 2nd Sem Grade by University",
            color_discrete_sequence=UNI_COLORS,
        )
        dark_layout(fig_ag, height=380)
        fig_ag.update_layout(showlegend=False)
        r2.plotly_chart(fig_ag, use_container_width=True)

    with tab_radar:
        metrics = ["Dropout_Rate", "Grad_Rate", "High_Risk", "Avg_Risk", "Avg_Grade"]
        radar_df = uni_agg[["University"] + metrics].copy()
        for m in metrics:
            mn, mx = radar_df[m].min(), radar_df[m].max()
            radar_df[m] = (radar_df[m] - mn) / (mx - mn + 1e-9)
        for m in ["Dropout_Rate", "High_Risk", "Avg_Risk"]:
            radar_df[m] = 1 - radar_df[m]

        labels = ["Low Dropout (↑=good)", "High Grad Rate", "Low High-Risk", "Low Avg Risk", "High Grade"]

        fig_radar_cmp = go.Figure()
        for i, row in radar_df.iterrows():
            vals = [row[m] for m in metrics]
            fig_radar_cmp.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    name=row["University"],
                    line_color=UNI_COLORS[i % len(UNI_COLORS)],
                    opacity=0.7,
                )
            )
        fig_radar_cmp.update_layout(
            polar=dict(bgcolor="#1a2035", radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor="#0f1117",
            font_color="#b0b8c8",
            title="University Performance Radar (normalised – higher = better)",
            height=520,
        )
        st.plotly_chart(fig_radar_cmp, use_container_width=True)

    with tab_trend:
        trend_all = (
            cmp_df.groupby(["University", "Enrollment_Year"])
            .agg(
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                Grad_Rate=("Predicted_Target", lambda x: (x == "Graduate").mean()),
                High_Risk=("Risk_Label", lambda x: (x == "High").mean()),
                Students=("Risk_Score", "count"),
            )
            .reset_index()
        )

        metric_sel = st.selectbox("Metric", ["Dropout_Rate", "Grad_Rate", "High_Risk", "Students"])

        fig_trend_all = px.line(
            trend_all,
            x="Enrollment_Year",
            y=metric_sel,
            color="University",
            markers=True,
            title=f"{metric_sel} by Enrollment Year – All Universities",
            color_discrete_sequence=UNI_COLORS,
        )
        dark_layout(fig_trend_all, height=440)
        st.plotly_chart(fig_trend_all, use_container_width=True)

        if {"Unemployment rate", "GDP", "Risk_Score"}.issubset(cmp_df.columns):
            st.caption("Macro-economic charts were moved to the dedicated Macro-Economic page.")

    with tab_table:
        display_tbl = uni_agg.copy()
        st.dataframe(
            display_tbl.style.format(
                {
                    "Dropout_Rate": "{:.1%}",
                    "Grad_Rate": "{:.1%}",
                    "Pending_Rate": "{:.1%}",
                    "High_Risk": "{:.1%}",
                    "Avg_Risk": "{:.1%}",
                    "Avg_Grade": "{:.2f}",
                    "Emirati_Pct": "{:.1%}",
                    "Expat_Pct": "{:.1%}",
                    "Abroad_Pct": "{:.1%}",
                }
            )
            .background_gradient(subset=["Dropout_Rate"], cmap="Reds")
            .background_gradient(subset=["Grad_Rate"], cmap="Greens"),
            use_container_width=True,
        )

        comp_data = uni_agg[["University", "Emirati_Pct", "Expat_Pct", "Abroad_Pct"]].melt(
            id_vars="University", var_name="Type", value_name="Pct"
        )
        comp_data["Type"] = comp_data["Type"].str.replace("_Pct", "")
        fig_comp = px.bar(
            comp_data,
            x="University",
            y="Pct",
            color="Type",
            color_discrete_map={"Emirati": "#60a5fa", "Expat": "#a78bfa", "Abroad": "#34d399"},
            barmode="stack",
            title="Student Type Composition by University",
            labels={"Pct": "Proportion"},
        )
        dark_layout(fig_comp, height=380)
        st.plotly_chart(fig_comp, use_container_width=True)

        if "Curricular units 2nd sem (grade)" in cmp_df.columns:
            fig_uni_grade = px.box(
                cmp_df,
                x="University",
                y="Curricular units 2nd sem (grade)",
                color="University",
                title="Grade Distribution by University",
                color_discrete_sequence=UNI_COLORS,
            )
            dark_layout(fig_uni_grade, height=420)
            st.plotly_chart(fig_uni_grade, use_container_width=True)

    with tab_scatter:
        fig_scat_cmp = px.scatter(
            uni_agg,
            x="Grad_Rate",
            y="Dropout_Rate",
            size="Students",
            color="University",
            text="University",
            title="Graduation Rate vs Dropout Rate (bubble = # students)",
            color_discrete_sequence=UNI_COLORS,
            labels={"Grad_Rate": "Graduation Rate", "Dropout_Rate": "Dropout Rate"},
        )
        fig_scat_cmp.update_traces(textposition="top center")
        dark_layout(fig_scat_cmp, height=500)
        st.plotly_chart(fig_scat_cmp, use_container_width=True)

        fig_scat2 = px.scatter(
            uni_agg,
            x="Avg_Grade",
            y="High_Risk",
            size="Students",
            color="University",
            text="University",
            title="Avg Grade vs High-Risk % (bubble = # students)",
            color_discrete_sequence=UNI_COLORS,
        )
        fig_scat2.update_traces(textposition="top center")
        dark_layout(fig_scat2, height=500)
        st.plotly_chart(fig_scat2, use_container_width=True)
