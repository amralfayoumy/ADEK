import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.constants import COLOR_MAP, RISK_MAP
from dashboard.ui import dark_layout, kpi


def render(df_full, display_outcome):
    st.markdown("# :material/flight_takeoff: Students Abroad")
    st.markdown(
        "Exclusively Emirati students studying outside the UAE — risk profile, performance, "
        "and comparisons against domestic Emiratis."
    )

    abroad_df = df_full[df_full["Student_Type"] == "Abroad"].copy()
    domestic_em = df_full[df_full["Student_Type"] == "Emirati"].copy()

    if abroad_df.empty:
        st.warning("No 'Abroad' students found in the current dataset.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, len(abroad_df), "Emiratis Abroad", "")
    kpi(c2, f"{(abroad_df['Predicted_Target']=='Dropout').mean()*100:.1f}%", "Dropout Rate", "", "#f87171")
    kpi(c3, f"{(abroad_df['Predicted_Target']=='Graduate').mean()*100:.1f}%", "Graduation Rate", "", "#34d399")
    kpi(c4, f"{(abroad_df['Risk_Label']=='High').mean()*100:.1f}%", "High-Risk %", "", "#fbbf24")
    st.markdown("<br>", unsafe_allow_html=True)

    tab_overview, tab_compare, tab_uni, tab_prog = st.tabs(
        [":material/dashboard: Overview", ":material/compare_arrows: vs Domestic Emiratis", ":material/account_balance: By University", ":material/school: By Program"]
    )

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pending Outcome %", f"{(abroad_df['Predicted_Target']=='Enrolled').mean()*100:.1f}%")
        c2.metric("Dropout %", f"{(abroad_df['Predicted_Target']=='Dropout').mean()*100:.1f}%")
        c3.metric("Graduation %", f"{(abroad_df['Predicted_Target']=='Graduate').mean()*100:.1f}%")

        r1, r2 = st.columns(2)
        risk_ab = abroad_df["Risk_Label"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
        risk_ab.columns = ["Risk", "Count"]
        fig_risk_ab = px.bar(
            risk_ab,
            x="Risk",
            y="Count",
            color="Risk",
            color_discrete_map=RISK_MAP,
            text="Count",
            title="Risk Level Distribution – Abroad",
        )
        dark_layout(fig_risk_ab, height=360)
        fig_risk_ab.update_traces(textposition="outside")
        r1.plotly_chart(fig_risk_ab, use_container_width=True)

        st.caption("Outcome distribution comparison is emphasized in the 'vs Domestic Emiratis' tab to reduce duplicate summaries.")

        drop_ab = abroad_df[abroad_df["Dropout_Reason"] != ""]
        if not drop_ab.empty:
            reason_ab = drop_ab["Dropout_Reason"].value_counts().reset_index()
            reason_ab.columns = ["Reason", "Count"]
            fig_reas = px.pie(
                reason_ab,
                names="Reason",
                values="Count",
                hole=0.45,
                title="Why Abroad Students Drop Out",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            dark_layout(fig_reas)
            st.plotly_chart(fig_reas, use_container_width=True)

        trend_ab = (
            abroad_df.groupby("Enrollment_Year")
            .agg(
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                Grad_Rate=("Predicted_Target", lambda x: (x == "Graduate").mean()),
                Count=("Risk_Score", "count"),
            )
            .reset_index()
        )
        fig_trend_ab = px.line(
            trend_ab,
            x="Enrollment_Year",
            y=["Dropout_Rate", "Grad_Rate"],
            markers=True,
            title="Abroad Students – Year-over-Year Trend",
            color_discrete_map={"Dropout_Rate": "#f87171", "Grad_Rate": "#34d399"},
        )
        dark_layout(fig_trend_ab, height=360)
        r2.plotly_chart(fig_trend_ab, use_container_width=True)

    with tab_compare:
        st.markdown("### Abroad Emiratis vs Domestic Emiratis")

        def ab_kpis(grp, label):
            return {
                "Group": label,
                "Count": len(grp),
                "Dropout Rate": f"{(grp['Predicted_Target']=='Dropout').mean()*100:.1f}%",
                "Grad Rate": f"{(grp['Predicted_Target']=='Graduate').mean()*100:.1f}%",
                "High Risk %": f"{(grp['Risk_Label']=='High').mean()*100:.1f}%",
                "Avg Risk Score": f"{grp['Risk_Score'].mean()*100:.1f}%",
                "Avg Grade": f"{grp['Curricular units 2nd sem (grade)'].mean():.2f}",
            }

        cmp = pd.DataFrame([
            ab_kpis(abroad_df, "Abroad"),
            ab_kpis(domestic_em, "Domestic Emirati"),
        ]).set_index("Group")
        st.dataframe(cmp, use_container_width=True)

        both = pd.concat(
            [
                abroad_df.assign(Group="Abroad"),
                domestic_em.assign(Group="Domestic Emirati"),
            ]
        )
        grp_colors = {"Abroad": "#34d399", "Domestic Emirati": "#60a5fa"}

        r1, r2 = st.columns(2)

        fig_v1 = px.violin(
            both,
            x="Group",
            y="Risk_Score",
            color="Group",
            color_discrete_map=grp_colors,
            box=True,
            title="Risk Score: Abroad vs Domestic",
            labels={"Risk_Score": "P(Dropout)"},
        )
        dark_layout(fig_v1, height=380)
        r1.plotly_chart(fig_v1, use_container_width=True)

        fig_v2 = px.violin(
            both,
            x="Group",
            y="Curricular units 2nd sem (grade)",
            color="Group",
            color_discrete_map=grp_colors,
            box=True,
            title="2nd Sem Grade: Abroad vs Domestic",
        )
        dark_layout(fig_v2, height=380)
        r2.plotly_chart(fig_v2, use_container_width=True)

        out_both = both.groupby(["Group", "Predicted_Target"]).size().reset_index(name="Count")
        out_both["Predicted_Target_Display"] = out_both["Predicted_Target"].map(display_outcome)
        fig_out_both = px.bar(
            out_both,
            x="Group",
            y="Count",
            color="Predicted_Target_Display",
            color_discrete_map=COLOR_MAP,
            barmode="stack",
            title="Predicted Outcomes: Abroad vs Domestic",
        )
        dark_layout(fig_out_both, height=380)
        st.plotly_chart(fig_out_both, use_container_width=True)

    with tab_uni:
        uni_ab = (
            abroad_df.groupby("University")
            .agg(
                Students=("Risk_Score", "count"),
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                Grad_Rate=("Predicted_Target", lambda x: (x == "Graduate").mean()),
                High_Risk=("Risk_Label", lambda x: (x == "High").mean()),
            )
            .reset_index()
            .sort_values("Dropout_Rate", ascending=False)
        )
        fig_uni_ab = px.bar(
            uni_ab,
            x="University",
            y=["Dropout_Rate", "Grad_Rate"],
            barmode="group",
            title="Abroad Student Outcomes by University",
            color_discrete_map={"Dropout_Rate": "#f87171", "Grad_Rate": "#34d399"},
        )
        dark_layout(fig_uni_ab, height=400)
        st.plotly_chart(fig_uni_ab, use_container_width=True)

        st.caption("Use University Comparison for side-by-side benchmarking with other cohorts and metrics.")

        st.dataframe(
            uni_ab.style.format({"Dropout_Rate": "{:.1%}", "Grad_Rate": "{:.1%}", "High_Risk": "{:.1%}"}),
            use_container_width=True,
        )

    with tab_prog:
        prog_ab = (
            abroad_df.groupby("Program")
            .agg(
                Students=("Risk_Score", "count"),
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                Avg_Grade=("Curricular units 2nd sem (grade)", "mean"),
            )
            .reset_index()
            .sort_values("Dropout_Rate", ascending=False)
        )
        fig_prog_ab = px.scatter(
            prog_ab,
            x="Avg_Grade",
            y="Dropout_Rate",
            size="Students",
            color="Dropout_Rate",
            color_continuous_scale="Reds",
            text="Program",
            title="Programs: Avg Grade vs Dropout Rate (bubble = #students)",
        )
        fig_prog_ab.update_traces(textposition="top center")
        dark_layout(fig_prog_ab, height=480)
        st.plotly_chart(fig_prog_ab, use_container_width=True)
