import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import get_trainer


def render(display_outcome, outcome_display_order):
    st.markdown("# :material/psychology: Predict New Student Risk")
    st.markdown("Enter a student's information to get an instant risk prediction.")

    mt = get_trainer()
    art, le_obj, feat_cols, thresholds = mt.load_artefacts()

    if art is None:
        st.error("Model not loaded.")
        st.stop()

    with st.form("predict_form"):
        st.markdown("<p class='section-header'>Demographics</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        age = c2.number_input("Age at Enrollment", 17, 70, 22)
        international = c3.selectbox("International", ["No", "Yes"])
        marital = c1.selectbox(
            "Marital Status",
            [1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Single",
                2: "Married",
                3: "Widower",
                4: "Divorced",
                5: "Facto union",
                6: "Legally separated",
            }[x],
        )
        displaced = c2.selectbox("Displaced", ["No", "Yes"])
        special_needs = c3.selectbox("Special Needs", ["No", "Yes"])

        st.markdown("<p class='section-header'>Financial</p>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        scholarship = f1.selectbox("Scholarship Holder", ["No", "Yes"])
        debtor = f2.selectbox("Debtor", ["No", "Yes"])
        fees_ok = f3.selectbox("Tuition Fees Up to Date", ["Yes", "No"])

        st.markdown("<p class='section-header'>Academic</p>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        admission_grade = a1.slider("Admission Grade", 0.0, 200.0, 130.0)
        prev_qual_grade = a2.slider("Previous Qualification Grade", 0.0, 200.0, 130.0)

        a3, a4 = st.columns(2)
        s1_enrolled = a3.number_input("1st Sem – Units Enrolled", 0, 30, 6)
        s1_approved = a4.number_input("1st Sem – Units Approved", 0, 30, 5)
        s1_grade = a3.number_input("1st Sem – Grade (0-20)", 0.0, 20.0, 12.0)
        s1_evals = a4.number_input("1st Sem – Evaluations", 0, 50, 6)
        s1_no_eval = a3.number_input("1st Sem – No Evaluations", 0, 30, 0)
        s1_credited = a4.number_input("1st Sem – Units Credited", 0, 30, 0)

        a5, a6 = st.columns(2)
        s2_enrolled = a5.number_input("2nd Sem – Units Enrolled", 0, 30, 6)
        s2_approved = a6.number_input("2nd Sem – Units Approved", 0, 30, 5)
        s2_grade = a5.number_input("2nd Sem – Grade (0-20)", 0.0, 20.0, 12.0)
        s2_evals = a6.number_input("2nd Sem – Evaluations", 0, 50, 6)
        s2_no_eval = a5.number_input("2nd Sem – No Evaluations", 0, 30, 0)
        s2_credited = a6.number_input("2nd Sem – Units Credited", 0, 30, 0)

        st.markdown("<p class='section-header'>Macro-Economic</p>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        unemployment = e1.number_input("Unemployment Rate (%)", 0.0, 20.0, 10.8)
        inflation = e2.number_input("Inflation Rate (%)", -5.0, 10.0, 1.4)
        gdp = e3.number_input("GDP", -5.0, 5.0, 1.74)

        submitted = st.form_submit_button(":material/auto_fix_high: Predict Risk", type="primary", use_container_width=True)

    if submitted:
        row_data = {c: 0 for c in feat_cols}
        row_data.update(
            {
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
            }
        )
        x_new = pd.DataFrame([row_data])[feat_cols]
        n_cls = 3
        stacked = np.zeros((1, n_cls * len(art["base_models"])))
        for m_idx, (_, model) in enumerate(art["base_models"]):
            probs = model.predict_proba(x_new)
            stacked[:, m_idx * n_cls : (m_idx + 1) * n_cls] = probs

        pred_encoded = art["meta_model"].predict(stacked)[0]
        pred_proba = art["meta_model"].predict_proba(stacked)[0]
        pred_label = le_obj.inverse_transform([pred_encoded])[0]
        pred_label_display = display_outcome(pred_label)
        cls_list = list(le_obj.classes_)
        p_dropout = pred_proba[cls_list.index("Dropout")]
        p_enrolled = pred_proba[cls_list.index("Enrolled")]
        p_grad = pred_proba[cls_list.index("Graduate")]
        q60 = thresholds["q60"]
        q75 = thresholds["q75"]
        risk_level = "High" if p_dropout > q75 else ("Medium" if p_dropout > q60 else "Low")
        risk_clr = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}[risk_level]

        st.markdown("---")
        st.markdown("## :material/analytics: Prediction Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(
            f'<div class="kpi-card"><p class="kpi-value" style="color:{risk_clr}">{p_dropout*100:.1f}%</p><p class="kpi-label">Dropout Risk</p></div>',
            unsafe_allow_html=True,
        )
        r2.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{pred_label_display}</p><p class="kpi-label">Predicted Outcome</p></div>',
            unsafe_allow_html=True,
        )
        r3.markdown(
            f'<div class="kpi-card"><p class="kpi-value" style="color:{risk_clr}">{risk_level}</p><p class="kpi-label">Risk Level</p></div>',
            unsafe_allow_html=True,
        )
        r4.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{p_grad*100:.1f}%</p><p class="kpi-label">P(Graduate)</p></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        fig_new = go.Figure(
            go.Bar(
                x=[p_dropout, p_enrolled, p_grad],
                y=outcome_display_order,
                orientation="h",
                marker_color=["#f87171", "#fbbf24", "#34d399"],
                text=[f"{v*100:.1f}%" for v in [p_dropout, p_enrolled, p_grad]],
                textposition="outside",
            )
        )
        fig_new.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font_color="#b0b8c8",
            xaxis=dict(range=[0, 1.1], tickformat=".0%"),
            height=220,
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_new, use_container_width=True)
