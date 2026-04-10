import os

import pandas as pd
import streamlit as st

DATA_CSV = "data.csv"
SCORES_CSV = "models/all_student_scores.csv"


@st.cache_resource(show_spinner=False)
def get_trainer():
    import model_trainer

    return model_trainer


def need_training():
    return not (
        os.path.exists("models/stacking_ensemble.joblib") and os.path.exists(SCORES_CSV)
    )


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    mt = get_trainer()
    return mt.load_scores()


def run_training(data_csv=DATA_CSV):
    mt = get_trainer()
    with st.spinner("🔄 Training ensemble model … (first run only – ~2-5 min)"):
        df = mt.train_and_save(data_csv)
    st.cache_data.clear()
    return df


def merge_enrichment_columns(df_full: pd.DataFrame, data_csv=DATA_CSV) -> pd.DataFrame:
    if "University" in df_full.columns:
        return df_full

    src = pd.read_csv(data_csv)
    src.columns = src.columns.str.strip()
    for col in [
        "University",
        "College",
        "Program",
        "Student_Type",
        "Enrollment_Year",
        "Dropout_Reason",
    ]:
        if col in src.columns:
            df_full[col] = src[col].values
    return df_full


def add_display_columns(df_full: pd.DataFrame, df: pd.DataFrame, display_outcome):
    for frame in (df_full, df):
        frame["Predicted_Target_Display"] = (
            frame["Predicted_Target"].map(display_outcome).fillna(frame["Predicted_Target"])
        )
        if "Target" in frame.columns:
            frame["Target_Display"] = frame["Target"].map(display_outcome).fillna(
                frame["Target"]
            )
    return df_full, df


def filter_dataframe(
    df_full: pd.DataFrame,
    sel_unis,
    sel_colleges,
    sel_programs,
    sel_stypes,
    risk_filter,
) -> pd.DataFrame:
    mask = (
        df_full["University"].isin(sel_unis)
        & df_full["College"].isin(sel_colleges)
        & df_full["Program"].isin(sel_programs)
        & df_full["Student_Type"].isin(sel_stypes)
        & df_full["Risk_Label"].isin(risk_filter)
    )
    df = df_full[mask].copy()
    df.index = range(len(df))
    return df
