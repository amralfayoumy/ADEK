import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.constants import COLOR_MAP, RISK_MAP, STYPE_MAP
from dashboard.ui import dark_layout, persist_streamlit_tabs


def render(df_full, display_outcome):
    st.markdown("# :material/groups: Emirati vs Expat Student Performance")
    st.markdown("Comparative analysis across all metrics — domestic Emiratis vs expatriate students.")

    ev_df = df_full[df_full["Student_Type"].isin(["Emirati", "Expat"])].copy()

    em = ev_df[ev_df["Student_Type"] == "Emirati"]
    ex = ev_df[ev_df["Student_Type"] == "Expat"]

    st.markdown("<p class='section-header'>Overview Comparison</p>", unsafe_allow_html=True)

    def grp_kpis(grp):
        return {
            "Count": len(grp),
            "Dropout Rate": f"{(grp['Predicted_Target']=='Dropout').mean()*100:.1f}%",
            "Grad Rate": f"{(grp['Predicted_Target']=='Graduate').mean()*100:.1f}%",
            "High Risk %": f"{(grp['Risk_Label']=='High').mean()*100:.1f}%",
            "Avg Risk Score": f"{grp['Risk_Score'].mean()*100:.1f}%",
            "Avg 2nd Sem Grade": f"{grp['Curricular units 2nd sem (grade)'].mean():.2f}",
            "Scholarship %": f"{grp['Scholarship holder'].mean()*100:.1f}%",
            "Debtor %": f"{grp['Debtor'].mean()*100:.1f}%",
        }

    cmp_df = pd.DataFrame([grp_kpis(em), grp_kpis(ex)], index=["Emirati", "Expat"]).T.astype(str)
    st.dataframe(cmp_df, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_options = [
        ":material/pie_chart: Outcomes",
        ":material/school: Academic",
        ":material/account_balance_wallet: Financial",
        ":material/account_balance: By University",
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_options)
    persist_streamlit_tabs("emirati_vs_expats_active_tab", tab_options)

    with tab1:
        r1, r2 = st.columns(2)

        outcome_data = ev_df.groupby(["Student_Type", "Predicted_Target"]).size().reset_index(name="Count")
        outcome_data["Predicted_Target_Display"] = outcome_data["Predicted_Target"].map(display_outcome)
        fig_out = px.bar(
            outcome_data,
            x="Student_Type",
            y="Count",
            color="Predicted_Target_Display",
            color_discrete_map=COLOR_MAP,
            barmode="stack",
            title="Outcome Distribution (Count)",
        )
        dark_layout(fig_out, height=360)
        r1.plotly_chart(fig_out, width="stretch")

        risk_grp = ev_df.groupby(["Student_Type", "Risk_Label"]).size().reset_index(name="Count")
        fig_risk_ev = px.bar(
            risk_grp,
            x="Student_Type",
            y="Count",
            color="Risk_Label",
            color_discrete_map=RISK_MAP,
            barmode="group",
            title="Risk Level Distribution",
        )
        dark_layout(fig_risk_ev, height=360)
        r2.plotly_chart(fig_risk_ev, width="stretch")

        fig_violin = px.violin(
            ev_df,
            x="Student_Type",
            y="Risk_Score",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            box=True,
            title="Dropout Risk Score Distribution",
            labels={"Risk_Score": "P(Dropout)"},
        )
        dark_layout(fig_violin, height=360)
        st.plotly_chart(fig_violin, width="stretch")

    with tab2:
        r1, r2 = st.columns(2)

        fig_grade = px.violin(
            ev_df,
            x="Student_Type",
            y="Curricular units 2nd sem (grade)",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            box=True,
            title="2nd Semester Grade Distribution",
        )
        dark_layout(fig_grade, height=360)
        r1.plotly_chart(fig_grade, width="stretch")

        fig_approved = px.histogram(
            ev_df,
            x="Curricular units 2nd sem (approved)",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            barmode="overlay",
            opacity=0.7,
            title="Units Approved – 2nd Semester",
        )
        dark_layout(fig_approved, height=360)
        r2.plotly_chart(fig_approved, width="stretch")

        st.caption("Program-level grade benchmarking is consolidated in College / Program Deep Dive.")

    with tab3:
        r1, r2 = st.columns(2)

        fin_metrics = (
            ev_df.groupby("Student_Type")
            .agg(Scholarship=("Scholarship holder", "mean"), Debtor=("Debtor", "mean"), Fees_OK=("Tuition fees up to date", "mean"))
            .reset_index()
            .melt(id_vars="Student_Type", var_name="Metric", value_name="Rate")
        )
        fig_fin = px.bar(
            fin_metrics,
            x="Metric",
            y="Rate",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            barmode="group",
            title="Financial Indicators Comparison",
            labels={"Rate": "Proportion"},
        )
        dark_layout(fig_fin, height=360)
        r1.plotly_chart(fig_fin, width="stretch")

        ev_df["Has Scholarship"] = ev_df["Scholarship holder"].map({1: "Yes", 0: "No"})
        fig_sch = px.box(
            ev_df,
            x="Has Scholarship",
            y="Risk_Score",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            title="Scholarship Impact on Dropout Risk",
        )
        dark_layout(fig_sch, height=360)
        r2.plotly_chart(fig_sch, width="stretch")

    with tab4:
        uni_ev = (
            ev_df.groupby(["University", "Student_Type"])
            .apply(
                lambda g: pd.Series(
                    {
                        "Dropout Rate": (g["Predicted_Target"] == "Dropout").mean(),
                        "Grad Rate": (g["Predicted_Target"] == "Graduate").mean(),
                        "High Risk %": (g["Risk_Label"] == "High").mean(),
                    }
                )
            )
            .reset_index()
        )
        fig_uni_ev = px.bar(
            uni_ev,
            x="University",
            y="Dropout Rate",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            barmode="group",
            title="Dropout Rate: Emirati vs Expat by University",
        )
        dark_layout(fig_uni_ev, height=380)
        st.plotly_chart(fig_uni_ev, width="stretch")

        fig_uni_g = px.bar(
            uni_ev,
            x="University",
            y="Grad Rate",
            color="Student_Type",
            color_discrete_map=STYPE_MAP,
            barmode="group",
            title="Graduation Rate: Emirati vs Expat by University",
        )
        dark_layout(fig_uni_g, height=380)
        st.plotly_chart(fig_uni_g, width="stretch")
