import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dashboard.constants import OUTCOME_DISPLAY_ORDER, OUTCOME_LABEL_MAP, OUTCOME_RAW_ORDER
from dashboard.ui import dark_layout


def render(df_full, get_trainer, display_outcome):
    st.markdown("# 📉 Model Performance Analysis")

    y_true = df_full["Target"]
    y_pred = df_full["Predicted_Target"]
    accuracy = accuracy_score(y_true, y_pred)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().rename(index=OUTCOME_LABEL_MAP)
        st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    with col2:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=OUTCOME_RAW_ORDER)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=OUTCOME_DISPLAY_ORDER,
            y=OUTCOME_DISPLAY_ORDER,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        dark_layout(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Meta-Learner Feature Importance")
    mt = get_trainer()
    art, le_obj, _feat_cols, _thresholds = mt.load_artefacts()
    if art is None:
        st.warning("Model artefacts are unavailable for feature-importance diagnostics.")
    else:
        meta = art["meta_model"]
        labels = []
        for bname, _ in art["base_models"]:
            for cls in le_obj.classes_:
                labels.append(f"{bname}→P({display_outcome(cls)})")
        importances = np.asarray(meta.feature_importances_)
        n = min(len(labels), len(importances))
        if n == 0:
            st.info("Meta-learner feature importances are not available in current artefacts.")
        else:
            imp2 = (
                pd.DataFrame({"Feature": labels[:n], "Importance": importances[:n]})
                .sort_values("Importance", ascending=True)
                .tail(15)
            )
            fig_imp = px.bar(
                imp2,
                y="Feature",
                x="Importance",
                orientation="h",
                title="Meta-Learner Feature Importance (Top 15)",
                color="Importance",
                color_continuous_scale="Blues",
            )
            dark_layout(fig_imp, height=460)
            st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "This page displays actual historical outcomes for model validation purposes. "
        "Pending is the dashboard display name for the model's internal Enrolled class."
    )
