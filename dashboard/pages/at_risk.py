import plotly.express as px
import streamlit as st

from dashboard.ui import dark_layout


def render(df, display_outcome):
    st.markdown("# 🚨 Early Warning System")

    tab_dropout, tab_engage = st.tabs(["🔴 Dropout Risk", "🟡 Low Engagement"])

    with tab_dropout:
        high_risk = df[df["Risk_Label"] == "High"].sort_values("Risk_Score", ascending=False)
        st.markdown(
            f"<p class='section-header'>Showing {len(high_risk)} high-risk students</p>",
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("High-Risk Students", len(high_risk))
        s2.metric("Avg Dropout Prob", f"{high_risk['Risk_Score'].mean()*100:.1f}%")
        s3.metric(
            "Universities Affected",
            high_risk["University"].nunique() if "University" in high_risk.columns else "N/A",
        )

        st.markdown("---")

        display_cols = [
            "University",
            "College",
            "Program",
            "Student_Type",
            "Predicted_Target",
            "Risk_Score",
            "Risk_Label",
            "Gender",
            "Age at enrollment",
            "Scholarship holder",
            "Debtor",
            "Tuition fees up to date",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Prob_Dropout",
            "Prob_Graduate",
        ]
        display_cols = [c for c in display_cols if c in high_risk.columns]
        risk_table = high_risk[display_cols].copy()
        if "Predicted_Target" in risk_table.columns:
            risk_table["Predicted_Target"] = (
                risk_table["Predicted_Target"].map(display_outcome).fillna(risk_table["Predicted_Target"])
            )

        st.dataframe(
            risk_table
            .rename(columns={"Predicted_Target": "Predicted", "Risk_Score": "P(Dropout)", "Risk_Label": "Risk"})
            .style.format({"P(Dropout)": "{:.2%}", "Prob_Dropout": "{:.2%}", "Prob_Graduate": "{:.2%}"})
            .background_gradient(subset=["P(Dropout)"], cmap="Reds"),
            use_container_width=True,
            height=480,
        )

        fig_uni_risk = px.bar(
            high_risk.groupby("University").size().reset_index(name="High Risk Count"),
            x="University",
            y="High Risk Count",
            color_discrete_sequence=["#f87171"],
            title="High-Risk Students per University",
        )
        dark_layout(fig_uni_risk, height=320)
        st.plotly_chart(fig_uni_risk, use_container_width=True)

    with tab_engage:
        low_eng = df[df["Engagement_Flag"] == "Low Engagement"]
        st.markdown(
            f"<p class='section-header'>{len(low_eng)} low-engagement students detected</p>",
            unsafe_allow_html=True,
        )
        eng_cols = [
            "University",
            "Program",
            "Student_Type",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Risk_Score",
            "Risk_Label",
        ]
        eng_cols = [c for c in eng_cols if c in low_eng.columns]
        st.dataframe(
            low_eng[eng_cols].style.format({"Risk_Score": "{:.2%}"}),
            use_container_width=True,
            height=420,
        )
