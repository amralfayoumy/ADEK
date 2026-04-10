import plotly.graph_objects as go
import streamlit as st

from dashboard.constants import STYPE_MAP
from dashboard.ui import dark_layout


def render(df, df_full, display_outcome, outcome_display_order):
    st.markdown("# :material/manage_search: Student Deep-Dive")

    sorted_df = df.sort_values("Risk_Score", ascending=False)
    col_sel, _ = st.columns([2, 1])

    student_idx = col_sel.selectbox(
        "Select student (row index)",
        sorted_df.index,
        format_func=lambda i: (
            f"Student #{i} | {sorted_df.loc[i,'University']} | "
            f"Risk: {sorted_df.loc[i,'Risk_Label']} "
            f"({sorted_df.loc[i,'Risk_Score']*100:.1f}%)"
            if "University" in sorted_df.columns
            else f"Student #{i} | Risk: {sorted_df.loc[i,'Risk_Label']} "
            f"({sorted_df.loc[i,'Risk_Score']*100:.1f}%)"
        ),
    )

    stu = sorted_df.loc[student_idx]
    stu_outcome = display_outcome(stu["Predicted_Target"])

    h1, h2, h3, h4 = st.columns(4)
    h1.markdown(
        f'<div class="kpi-card"><p class="kpi-value">{stu["Risk_Score"]*100:.1f}%</p><p class="kpi-label">Dropout Risk</p></div>',
        unsafe_allow_html=True,
    )
    h2.markdown(
        f'<div class="kpi-card"><p class="kpi-value">{stu_outcome}</p><p class="kpi-label">Model Prediction</p></div>',
        unsafe_allow_html=True,
    )
    risk_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}[str(stu["Risk_Label"])]
    h3.markdown(
        f'<div class="kpi-card"><p class="kpi-value" style="color:{risk_color}">{stu["Risk_Label"]}</p><p class="kpi-label">Risk Level</p></div>',
        unsafe_allow_html=True,
    )
    stype = stu.get("Student_Type", "N/A")
    stype_color = STYPE_MAP.get(stype, "#64748b")
    h4.markdown(
        f'<div class="kpi-card"><p class="kpi-value" style="color:{stype_color}">{stype}</p><p class="kpi-label">Student Type</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    fig_prob = go.Figure(
        go.Bar(
            x=[stu["Prob_Dropout"], stu["Prob_Enrolled"], stu["Prob_Graduate"]],
            y=outcome_display_order,
            orientation="h",
            marker_color=["#f87171", "#fbbf24", "#34d399"],
            text=[f"{v*100:.1f}%" for v in [stu["Prob_Dropout"], stu["Prob_Enrolled"], stu["Prob_Graduate"]]],
            textposition="outside",
        )
    )
    fig_prob.update_layout(
        title="Prediction Probabilities",
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
        margin=dict(t=40, b=10),
    )
    dark_layout(fig_prob, height=200)
    st.plotly_chart(fig_prob, width="stretch")

    st.markdown("<p class='section-header'>Student Profile</p>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**Demographics**")
        st.markdown(f"- Gender: {'Male' if stu.get('Gender')==1 else 'Female'}")
        st.markdown(f"- Age at enrollment: {stu.get('Age at enrollment','N/A')}")
        st.markdown(f"- International: {'Yes' if stu.get('International')==1 else 'No'}")
        st.markdown(f"- University: {stu.get('University','N/A')}")
        st.markdown(f"- College: {stu.get('College','N/A')}")
        st.markdown(f"- Program: {stu.get('Program','N/A')}")
        st.markdown(f"- Student Type: {stu.get('Student_Type','N/A')}")
        st.markdown(f"- Enrollment Year: {stu.get('Enrollment_Year','N/A')}")

    with p2:
        st.markdown("**Financial**")
        st.markdown(f"- Scholarship: {':material/check_circle:' if stu.get('Scholarship holder')==1 else ':material/cancel:'}")
        st.markdown(f"- Debtor: {':material/warning: Yes' if stu.get('Debtor')==1 else ':material/check_circle: No'}")
        st.markdown(f"- Fees up to date: {':material/check_circle:' if stu.get('Tuition fees up to date')==1 else ':material/cancel:'}")
        st.markdown(f"- Displaced: {'Yes' if stu.get('Displaced')==1 else 'No'}")

    with p3:
        st.markdown("**Academic (2nd Semester)**")
        st.markdown(f"- Units enrolled: {stu.get('Curricular units 2nd sem (enrolled)','N/A')}")
        st.markdown(f"- Units approved: {stu.get('Curricular units 2nd sem (approved)','N/A')}")
        st.markdown(f"- Grade: {stu.get('Curricular units 2nd sem (grade)',0):.2f}")
        st.markdown(f"- Evaluations: {stu.get('Curricular units 2nd sem (evaluations)','N/A')}")

    radar_features = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Admission grade",
        "Previous qualification (grade)",
    ]
    radar_features = [f for f in radar_features if f in df_full.columns]
    stu_norm, avg_norm = [], []
    for feat in radar_features:
        col_min, col_max = df_full[feat].min(), df_full[feat].max()
        rng_v = col_max - col_min or 1
        stu_norm.append((stu[feat] - col_min) / rng_v)
        avg_norm.append((df_full[feat].mean() - col_min) / rng_v)

    labels_short = [f.replace("Curricular units ", "").replace(" sem ", "S").replace("(", "").replace(")", "") for f in radar_features]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=stu_norm + [stu_norm[0]],
            theta=labels_short + [labels_short[0]],
            fill="toself",
            name="Student",
            line_color="#60a5fa",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=avg_norm + [avg_norm[0]],
            theta=labels_short + [labels_short[0]],
            fill="toself",
            name="Average",
            line_color="#a3e635",
            opacity=0.5,
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Student vs Cohort Average",
        height=380,
    )
    dark_layout(fig_radar)
    st.plotly_chart(fig_radar, width="stretch")

    st.markdown("<p class='section-header'>🤖 AI-Suggested Interventions</p>", unsafe_allow_html=True)

    def rule_interventions(s):
        tips = []
        if s.get("Debtor") == 1:
            tips.append(("💰 Financial Aid Referral", "Student has outstanding debt. Connect with the financial aid office for payment plans or emergency grants.", "High Priority"))
        if s.get("Tuition fees up to date") == 0:
            tips.append(("💳 Tuition Assistance", "Tuition fees are not current. Risk of administrative withdrawal. Immediate intervention required.", "High Priority"))
        if s.get("Curricular units 2nd sem (approved)", 5) == 0:
            tips.append(("📚 Academic Probation Review", "Zero units approved in the 2nd semester signals serious academic difficulty. Schedule an urgent academic counselling session.", "High Priority"))
        if s.get("Curricular units 2nd sem (grade)", 10) < 8 and s.get("Curricular units 2nd sem (grade)", 10) > 0:
            tips.append(("📖 Tutoring Programme", f"2nd semester grade is {s.get('Curricular units 2nd sem (grade)',0):.1f} - below passing threshold. Enroll in tutoring or peer-learning groups.", "Medium Priority"))
        if s.get("Scholarship holder") == 0 and s.get("Debtor") == 1:
            tips.append(("🎓 Scholarship Application", "Student is in debt without scholarship support. Advise on available merit/need-based scholarships.", "Medium Priority"))
        if s.get("Age at enrollment", 20) > 30:
            tips.append(("🌐 Adult Learner Support", "Mature student who may benefit from flexible scheduling, online resources, or mentoring from alumni.", "Low Priority"))
        if s.get("International") == 1:
            tips.append(("🌍 International Student Office", "International student - ensure visa status, housing, and language support are in order.", "Low Priority"))
        if s.get("Student_Type") == "Abroad":
            tips.append(("✈️ Abroad Student Liaison", "Emirati student studying abroad - ensure cultural adjustment support, UAE scholarship compliance, and regular check-ins.", "Medium Priority"))
        if not tips:
            tips.append(("✅ On Track", "No immediate red flags detected. Continue monitoring and provide encouragement.", "Informational"))
        return tips

    pri_color = {"High Priority": "#f87171", "Medium Priority": "#fbbf24", "Low Priority": "#34d399", "Informational": "#60a5fa"}
    for title_t, body, priority in rule_interventions(stu):
        clr = pri_color.get(priority, "#64748b")
        st.markdown(
            f'<div class="intervention-card">'
            f'<span class="intervention-title">{title_t}</span>'
            f'<span style="float:right;font-size:0.75rem;color:{clr};font-weight:600">{priority}</span>'
            f'<br><span style="font-size:0.9rem;color:var(--text-color);opacity:0.88">{body}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
