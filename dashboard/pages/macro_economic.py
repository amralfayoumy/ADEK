import plotly.express as px
import streamlit as st

from dashboard.constants import COLOR_MAP, HAS_STATSMODELS
from dashboard.ui import dark_layout


def render(df):
    st.markdown("# :material/public: Macro-Economic Analysis")
    st.markdown("How unemployment, inflation, and GDP relate to predicted dropout risk.")

    trendline_mode = "ols" if HAS_STATSMODELS else None
    if not HAS_STATSMODELS:
        st.info("Install statsmodels to enable OLS trendlines in macro-economic charts.")

    m1, m2 = st.columns(2)
    fig_unemp = px.scatter(
        df,
        x="Unemployment rate",
        y="Risk_Score",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        opacity=0.55,
        trendline=trendline_mode,
        title="Unemployment Rate vs Dropout Risk",
    )
    dark_layout(fig_unemp, height=380)
    m1.plotly_chart(fig_unemp, use_container_width=True)

    fig_gdp = px.scatter(
        df,
        x="GDP",
        y="Risk_Score",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        opacity=0.55,
        trendline=trendline_mode,
        title="GDP vs Dropout Risk",
    )
    dark_layout(fig_gdp, height=380)
    m2.plotly_chart(fig_gdp, use_container_width=True)

    fig_infl = px.scatter(
        df,
        x="Inflation rate",
        y="Risk_Score",
        color="Predicted_Target_Display",
        color_discrete_map=COLOR_MAP,
        opacity=0.55,
        trendline=trendline_mode,
        title="Inflation Rate vs Dropout Risk",
    )
    dark_layout(fig_infl, height=380)
    st.plotly_chart(fig_infl, use_container_width=True)
