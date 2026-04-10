import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.constants import COLOR_MAP, RISK_MAP, STYPE_MAP
from dashboard.ui import dark_layout, kpi


def render(df_full):
    st.markdown("# :material/account_balance: University Deep Dive")

    uni_list = sorted(df_full["University"].dropna().unique().tolist())
    sel_uni = st.selectbox("Select University", uni_list)

    udf = df_full[df_full["University"] == sel_uni].copy()

    total_u = len(udf)
    dropout_r = (udf["Predicted_Target"] == "Dropout").mean()
    grad_r = (udf["Predicted_Target"] == "Graduate").mean()
    high_r = (udf["Risk_Label"] == "High").mean()
    enrolled_r = (udf["Predicted_Target"] == "Enrolled").mean()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, total_u, "Total Students", "")
    kpi(c2, f"{dropout_r*100:.1f}%", "Dropout Rate", "", "#f87171")
    kpi(c3, f"{grad_r*100:.1f}%", "Graduation Rate", "", "#34d399")
    kpi(c4, f"{high_r*100:.1f}%", "High-Risk Students", "", "#fbbf24")
    kpi(c5, f"{enrolled_r*100:.1f}%", "Pending Outcome Rate", "")
    st.markdown("<br>", unsafe_allow_html=True)

    tab_risk, tab_funnel, tab_reasons, tab_trend = st.tabs(
        [":material/stacked_bar_chart: Risk Tiers", ":material/filter_alt: Enrollment Funnel", ":material/help: Dropout Reasons", ":material/trending_up: Year Trend"]
    )

    with tab_risk:
        r1, r2 = st.columns(2)

        risk_cnt = udf["Risk_Label"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
        risk_cnt.columns = ["Risk", "Count"]
        fig_risk_u = px.bar(
            risk_cnt,
            x="Risk",
            y="Count",
            color="Risk",
            color_discrete_map=RISK_MAP,
            text="Count",
            title=f"Risk Level Distribution – {sel_uni}",
        )
        dark_layout(fig_risk_u, height=340)
        fig_risk_u.update_traces(textposition="outside")
        r1.plotly_chart(fig_risk_u, use_container_width=True)

        coll_risk = (
            udf.groupby("College")
            .apply(
                lambda g: pd.Series(
                    {
                        "Dropout Rate": (g["Predicted_Target"] == "Dropout").mean(),
                        "High Risk %": (g["Risk_Label"] == "High").mean(),
                        "Students": len(g),
                    }
                )
            )
            .reset_index()
        )
        fig_coll = px.bar(
            coll_risk,
            x="College",
            y=["Dropout Rate", "High Risk %"],
            barmode="group",
            title="Dropout & High-Risk Rate by College",
            color_discrete_map={"Dropout Rate": "#f87171", "High Risk %": "#fbbf24"},
        )
        dark_layout(fig_coll, height=340)
        r2.plotly_chart(fig_coll, use_container_width=True)

        r3, r4 = st.columns(2)

        if "Student_Type" in udf.columns and not udf.empty:
            stype_split = udf["Student_Type"].value_counts().reset_index()
            stype_split.columns = ["Student_Type", "Count"]
            fig_stype = px.pie(
                stype_split,
                names="Student_Type",
                values="Count",
                color="Student_Type",
                color_discrete_map=STYPE_MAP,
                hole=0.5,
                title="Student Type Composition",
            )
            dark_layout(fig_stype, height=320)
            r3.plotly_chart(fig_stype, use_container_width=True)
        else:
            r3.info("Student-type data is unavailable for the selected filters.")

        fig_hist_u = px.histogram(
            udf,
            x="Risk_Score",
            color="Predicted_Target_Display",
            color_discrete_map=COLOR_MAP,
            nbins=30,
            title="Dropout Probability Distribution",
            barmode="overlay",
            opacity=0.75,
        )
        dark_layout(fig_hist_u, height=320)
        r4.plotly_chart(fig_hist_u, use_container_width=True)

    with tab_funnel:
        years = sorted(udf["Enrollment_Year"].dropna().unique())

        funnel_rows = []
        for yr in years:
            cohort = udf[udf["Enrollment_Year"] == yr]
            cohort_size = len(cohort)
            pending_count = (cohort["Predicted_Target"] == "Enrolled").sum()
            graduated = (cohort["Predicted_Target"] == "Graduate").sum()
            dropped = (cohort["Predicted_Target"] == "Dropout").sum()
            funnel_rows.append(
                {
                    "Year": str(yr),
                    "Cohort Size": cohort_size,
                    "Graduated": graduated,
                    "Dropped Out": dropped,
                    "Pending": pending_count,
                }
            )

        funnel_df = pd.DataFrame(funnel_rows)

        fig_funnel = go.Figure()
        fig_funnel.add_trace(
            go.Bar(name="Cohort Size", x=funnel_df["Year"], y=funnel_df["Cohort Size"], marker_color="#60a5fa")
        )
        fig_funnel.add_trace(
            go.Bar(name="Graduated", x=funnel_df["Year"], y=funnel_df["Graduated"], marker_color="#34d399")
        )
        fig_funnel.add_trace(
            go.Bar(name="Dropped Out", x=funnel_df["Year"], y=funnel_df["Dropped Out"], marker_color="#f87171")
        )
        fig_funnel.add_trace(
            go.Bar(name="Pending", x=funnel_df["Year"], y=funnel_df["Pending"], marker_color="#fbbf24")
        )
        fig_funnel.update_layout(
            barmode="group",
            title=f"Cohort Funnel by Enrollment Year – {sel_uni}",
            xaxis_title="Enrollment Year",
            yaxis_title="Students",
        )
        dark_layout(fig_funnel, height=420)
        st.plotly_chart(fig_funnel, use_container_width=True)

        cohort_n = len(udf)
        outcome_counts = (
            udf["Predicted_Target"]
            .value_counts()
            .reindex(["Dropout", "Enrolled", "Graduate"], fill_value=0)
        )
        dropped_n = int(outcome_counts["Dropout"])
        pending_n = int(outcome_counts["Enrolled"])
        graduated_n = int(outcome_counts["Graduate"])

        reason_counts = (
            udf.loc[udf["Predicted_Target"] == "Dropout", "Dropout_Reason"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Unspecified Reason")
            .value_counts()
        )

        reason_labels = [f"Reason: {r}" for r in reason_counts.index.tolist()]
        reason_values = [int(v) for v in reason_counts.tolist()]
        if not reason_labels:
            reason_labels = ["Reason: No predicted dropouts"]
            reason_values = [0]

        nodes = [
            "Cohort",
            "Predicted Dropout",
            "Predicted Pending",
            "Predicted Graduate",
            *reason_labels,
        ]

        sources = [0, 0, 0]
        targets = [1, 2, 3]
        values = [dropped_n, pending_n, graduated_n]

        for i, v in enumerate(reason_values):
            sources.append(1)
            targets.append(4 + i)
            values.append(v)

        node_colors = ["#60a5fa", "#f87171", "#fbbf24", "#34d399"] + ["#fb7185"] * len(reason_labels)
        link_colors = [
            "rgba(248,113,113,0.20)",
            "rgba(251,191,36,0.20)",
            "rgba(52,211,153,0.20)",
        ] + ["rgba(248,113,113,0.16)"] * len(reason_labels)

        fig_sk = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=nodes,
                    pad=18,
                    thickness=18,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        )
        fig_sk.update_layout(title=f"Overall Student Journey Sankey – {sel_uni}")
        dark_layout(fig_sk, height=430)
        st.plotly_chart(fig_sk, use_container_width=True)

    with tab_reasons:
        dropout_df = udf[(udf["Dropout_Reason"] != "") & (udf["Dropout_Reason"].notna())].copy()
        if dropout_df.empty:
            st.info("No dropout reason data for this university in the current filter.")
        else:
            r1, r2 = st.columns(2)

            reason_cnt = dropout_df["Dropout_Reason"].value_counts().reset_index()
            reason_cnt.columns = ["Reason", "Count"]
            fig_reason = px.pie(
                reason_cnt,
                names="Reason",
                values="Count",
                hole=0.45,
                title=f"Why Students Drop Out – {sel_uni}",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            dark_layout(fig_reason)
            r1.plotly_chart(fig_reason, use_container_width=True)

            reason_stype = dropout_df.groupby(["Dropout_Reason", "Student_Type"]).size().reset_index(name="Count")
            fig_rs = px.bar(
                reason_stype,
                x="Dropout_Reason",
                y="Count",
                color="Student_Type",
                color_discrete_map=STYPE_MAP,
                title="Dropout Reasons by Student Type",
                barmode="stack",
            )
            dark_layout(fig_rs, height=380)
            fig_rs.update_layout(xaxis_tickangle=-25)
            r2.plotly_chart(fig_rs, use_container_width=True)

            st.caption(
                "Dropout reasons by college/program are consolidated in College / Program Deep Dive for operational planning."
            )

    with tab_trend:
        trend = (
            udf.groupby("Enrollment_Year")
            .agg(
                Dropout_Rate=("Predicted_Target", lambda x: (x == "Dropout").mean()),
                Grad_Rate=("Predicted_Target", lambda x: (x == "Graduate").mean()),
                High_Risk=("Risk_Label", lambda x: (x == "High").mean()),
                Students=("Risk_Score", "count"),
            )
            .reset_index()
        )
        fig_trend = px.line(
            trend,
            x="Enrollment_Year",
            y=["Dropout_Rate", "Grad_Rate", "High_Risk"],
            markers=True,
            title=f"Year-over-Year Trends – {sel_uni}",
            color_discrete_map={"Dropout_Rate": "#f87171", "Grad_Rate": "#34d399", "High_Risk": "#fbbf24"},
            labels={"value": "Rate", "variable": "Metric"},
        )
        dark_layout(fig_trend, height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
