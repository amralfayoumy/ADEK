import plotly.express as px
import streamlit as st

from dashboard.feature_decoder import decode_dataframe_features
from dashboard.ui import dark_layout


def _financial_risk_level(row):
    debtor = row.get("Debtor")
    fees_up_to_date = row.get("Tuition fees up to date")
    scholarship = row.get("Scholarship holder")

    if debtor == 1 and fees_up_to_date == 0:
        return "Critical"
    if fees_up_to_date == 0:
        return "High"
    if debtor == 1 and scholarship == 0:
        return "Elevated"
    if debtor == 1:
        return "Medium"
    return "Low"


def _academic_alert(row):
    approved = row.get("Curricular units 2nd sem (approved)")
    grade = row.get("Curricular units 2nd sem (grade)")

    if approved == 0:
        return "No approvals"
    if isinstance(grade, (int, float)) and grade == grade and grade < 10:
        return "Low grade"
    if isinstance(approved, (int, float)) and approved <= 2:
        return "Low throughput"
    return "Monitor"


def render(df, display_outcome):
    st.markdown("# :material/warning: Early Warning System")

    tab_options = [":material/error: Dropout Risk", ":material/sentiment_dissatisfied: Low Engagement"]
    tabs_key = f"at_risk_tabs_{st.session_state.get('ui_tab_reset_nonce', 0)}"
    tab_dropout, tab_engage = st.tabs(tab_options, default=tab_options[0], key=tabs_key)

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

        risk_table = high_risk.copy()
        risk_table["Student ID"] = risk_table.index

        if "Predicted_Target" in risk_table.columns:
            risk_table["Predicted_Target"] = (
                risk_table["Predicted_Target"].map(display_outcome).fillna(risk_table["Predicted_Target"])
            )

        if {
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (enrolled)",
        }.issubset(risk_table.columns):
            enrolled = risk_table["Curricular units 2nd sem (enrolled)"]
            approved = risk_table["Curricular units 2nd sem (approved)"]
            risk_table["2nd Sem Progress"] = approved.div(enrolled.where(enrolled > 0))

        if {"Prob_Dropout", "Prob_Enrolled", "Prob_Graduate"}.issubset(risk_table.columns):
            next_best = risk_table[["Prob_Enrolled", "Prob_Graduate"]].max(axis=1)
            risk_table["Confidence Gap"] = risk_table["Prob_Dropout"] - next_best

        if {
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
        }.issubset(risk_table.columns):
            risk_table["Financial Risk"] = risk_table.apply(_financial_risk_level, axis=1)

        if {
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
        }.issubset(risk_table.columns):
            risk_table["Academic Alert"] = risk_table.apply(_academic_alert, axis=1)

        view_cols = [
            "Student ID",
            "University",
            "College",
            "Program",
            "Student_Type",
            "Predicted_Target",
            "Risk_Label",
            "Risk_Score",
            "Confidence Gap",
            "2nd Sem Progress",
            "Curricular units 2nd sem (grade)",
            "Financial Risk",
            "Academic Alert",
            "Gender",
            "Age at enrollment",
        ]
        view_cols = [c for c in view_cols if c in risk_table.columns]

        risk_view = risk_table[view_cols].copy()
        risk_view = decode_dataframe_features(risk_view)

        risk_view = risk_view.rename(
            columns={
                "Student_Type": "Student Type",
                "Predicted_Target": "Predicted",
                "Risk_Label": "Risk",
                "Risk_Score": "P(Dropout)",
                "Curricular units 2nd sem (grade)": "2nd Sem Grade",
            }
        )

        format_map = {}
        if "P(Dropout)" in risk_view.columns:
            format_map["P(Dropout)"] = "{:.2%}"
        if "Confidence Gap" in risk_view.columns:
            format_map["Confidence Gap"] = "{:+.2%}"
        if "2nd Sem Progress" in risk_view.columns:
            format_map["2nd Sem Progress"] = "{:.1%}"
        if "2nd Sem Grade" in risk_view.columns:
            format_map["2nd Sem Grade"] = "{:.2f}"

        risk_styler = risk_view.style
        if format_map:
            risk_styler = risk_styler.format(format_map)
        if "P(Dropout)" in risk_view.columns:
            risk_styler = risk_styler.background_gradient(subset=["P(Dropout)"], cmap="Reds")

        st.dataframe(
            risk_styler,
            width="stretch",
            height=480,
            hide_index=True,
        )

        fig_uni_risk = px.bar(
            high_risk.groupby("University").size().reset_index(name="High Risk Count"),
            x="University",
            y="High Risk Count",
            color_discrete_sequence=["#f87171"],
            title="High-Risk Students per University",
        )
        dark_layout(fig_uni_risk, height=320)
        st.plotly_chart(fig_uni_risk, width="stretch")

    with tab_engage:
        low_eng = df[df["Engagement_Flag"] == "Low Engagement"]
        st.markdown(
            f"<p class='section-header'>{len(low_eng)} low-engagement students detected</p>",
            unsafe_allow_html=True,
        )
        low_eng_table = low_eng.copy()
        low_eng_table["Student ID"] = low_eng_table.index

        if {
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (enrolled)",
        }.issubset(low_eng_table.columns):
            enrolled = low_eng_table["Curricular units 2nd sem (enrolled)"]
            approved = low_eng_table["Curricular units 2nd sem (approved)"]
            low_eng_table["2nd Sem Progress"] = approved.div(enrolled.where(enrolled > 0))

        if {
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
        }.issubset(low_eng_table.columns):
            low_eng_table["Financial Risk"] = low_eng_table.apply(_financial_risk_level, axis=1)

        if {
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
        }.issubset(low_eng_table.columns):
            low_eng_table["Academic Alert"] = low_eng_table.apply(_academic_alert, axis=1)

        if "Predicted_Target" in low_eng_table.columns:
            low_eng_table["Predicted_Target"] = (
                low_eng_table["Predicted_Target"].map(display_outcome).fillna(low_eng_table["Predicted_Target"])
            )

        eng_view_cols = [
            "Student ID",
            "University",
            "College",
            "Program",
            "Student_Type",
            "Predicted_Target",
            "Risk_Label",
            "Risk_Score",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "2nd Sem Progress",
            "Curricular units 2nd sem (grade)",
            "Financial Risk",
            "Academic Alert",
        ]
        eng_view_cols = [c for c in eng_view_cols if c in low_eng_table.columns]
        low_eng_table = low_eng_table[eng_view_cols].copy()

        low_eng_table = decode_dataframe_features(low_eng_table)
        low_eng_table = low_eng_table.rename(
            columns={
                "Student_Type": "Student Type",
                "Predicted_Target": "Predicted",
                "Risk_Score": "P(Dropout)",
                "Risk_Label": "Risk",
                "Curricular units 2nd sem (enrolled)": "2nd Sem Enrolled",
                "Curricular units 2nd sem (approved)": "2nd Sem Approved",
                "Curricular units 2nd sem (grade)": "2nd Sem Grade",
            }
        )

        format_map = {}
        if "P(Dropout)" in low_eng_table.columns:
            format_map["P(Dropout)"] = "{:.2%}"
        if "2nd Sem Progress" in low_eng_table.columns:
            format_map["2nd Sem Progress"] = "{:.1%}"
        if "2nd Sem Grade" in low_eng_table.columns:
            format_map["2nd Sem Grade"] = "{:.2f}"

        low_eng_styler = low_eng_table.style
        if format_map:
            low_eng_styler = low_eng_styler.format(format_map)
        if "P(Dropout)" in low_eng_table.columns:
            low_eng_styler = low_eng_styler.background_gradient(subset=["P(Dropout)"], cmap="Reds")

        st.dataframe(
            low_eng_styler,
            width="stretch",
            height=420,
            hide_index=True,
        )
