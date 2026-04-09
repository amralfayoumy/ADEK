"""
model_trainer.py
Trains the stacking ensemble (XGBoost + LightGBM + CatBoost → XGBoost meta-learner)
from the notebook and persists the fitted artefacts to disk.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
MODEL_PATH = "models/stacking_ensemble.joblib"
SCORES_PATH = "models/all_student_scores.csv"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"
THRESHOLDS_PATH = "models/risk_thresholds.joblib"


# ── Hyperparameters from notebook (Optuna-tuned) ──────────────────────────────

XGB_PARAMS = dict(
    n_estimators=929, alpha=2.287466581490129e-05, subsample=0.8766675651018592,
    colsample_bytree=0.288332829334817, max_depth=8, min_child_weight=6,
    learning_rate=0.024083411832750343, gamma=0.001816649055813574,
    random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="mlogloss",
)

LGBM_PARAMS = dict(
    n_estimators=1894, max_depth=23, learning_rate=0.024309983270196903,
    min_data_in_leaf=27, subsample=0.40065361124232945, max_bin=267,
    feature_fraction=0.1326832138080814, random_state=RANDOM_STATE, verbose=-1,
)

CATB_PARAMS = dict(
    colsample_bylevel=0.6383474716497279, learning_rate=0.09475494290429642,
    random_strength=0.07771221926568195, max_bin=490, depth=5, l2_leaf_reg=5,
    boosting_type="Plain", bootstrap_type="Bernoulli", subsample=0.8429457747642737,
    random_state=RANDOM_STATE, logging_level="Silent",
)

META_PARAMS = dict(
    n_estimators=47, alpha=6.422755620546236e-05, subsample=0.8452333586225941,
    colsample_bytree=0.7651776055349394, max_depth=3, min_child_weight=8,
    learning_rate=0.011014344390319484, gamma=6.1495867050966066e-06,
    random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="mlogloss",
)

BASE_MODELS = [
    ("xgb",  XGBClassifier(**XGB_PARAMS)),
    ("lgbm", LGBMClassifier(**LGBM_PARAMS)),
    ("catb", CatBoostClassifier(**CATB_PARAMS)),
]


# ── Core training helpers ─────────────────────────────────────────────────────

def _fit_base_models_cv(X: pd.DataFrame, y: pd.Series, cv):
    """
    Returns OOF probability matrix (n_samples × 9) and
    a list of fully-retrained base models (fitted on all data).
    """
    n_classes = 3
    n_models   = len(BASE_MODELS)
    oof_probs  = np.zeros((len(X), n_classes * n_models))
    fitted     = []

    for m_idx, (name, model) in enumerate(BASE_MODELS):
        print(f"  Training {name} ...")
        fold_probs = np.zeros((len(X), n_classes))

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
            Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
            Xval     = X.iloc[val_idx]
            clone    = model.__class__(**model.get_params())
            clone.fit(Xtr, ytr)
            fold_probs[val_idx] = clone.predict_proba(Xval)

        # full refit
        full_model = model.__class__(**model.get_params())
        full_model.fit(X, y)
        fitted.append((name, full_model))

        col_start = m_idx * n_classes
        oof_probs[:, col_start : col_start + n_classes] = fold_probs
        acc = accuracy_score(y, np.argmax(fold_probs, axis=1))
        print(f"    OOF accuracy: {acc:.4f}")

    return oof_probs, fitted


def train_and_save(data_path: str = "data.csv"):
    """
    Full training pipeline. Saves model artefacts under models/.
    Returns the complete scored DataFrame.
    """
    os.makedirs("models", exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    df = pd.read_csv(data_path)

    # The UCI CSV uses ';' as separator for some versions
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, sep=";")

    # Normalise column names
    df.columns = df.columns.str.strip()

    if "Target" not in df.columns:
        raise ValueError("'Target' column not found in the CSV.")

    feature_cols = [c for c in df.columns if c != "Target"]
    X = df[feature_cols].copy()
    y_raw = df["Target"].copy()

    # ── Encode target ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = pd.Series(le.fit_transform(y_raw), name="Target")
    # Ensure class order: Dropout=0, Enrolled=1, Graduate=2
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nTraining base models (5-fold CV) …")
    oof_probs, fitted_base = _fit_base_models_cv(X, y, cv)

    # ── Meta-learner ──────────────────────────────────────────────────────────
    print("\nTraining meta-learner …")
    meta_model = XGBClassifier(**META_PARAMS)
    meta_model.fit(oof_probs, y)
    meta_preds = meta_model.predict(oof_probs)
    print(f"  Meta-model accuracy (train): {accuracy_score(y, meta_preds):.4f}")

    # ── Persist artefacts ─────────────────────────────────────────────────────
    artefacts = {
        "base_models": fitted_base,
        "meta_model":  meta_model,
    }
    joblib.dump(artefacts,    MODEL_PATH)
    joblib.dump(le,           LABEL_ENCODER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    print(f"\nModels saved to {MODEL_PATH}")

    # ── Score entire dataset ──────────────────────────────────────────────────
    scored_df = score_dataset(df, artefacts, le, feature_cols)
    scored_df.to_csv(SCORES_PATH, index=False)
    print(f"Scores saved to {SCORES_PATH}")

    return scored_df


def score_dataset(df: pd.DataFrame, artefacts: dict, le, feature_cols: list):
    """Score every row and attach predictions + probabilities."""
    X = df[feature_cols].copy()

    base_models = artefacts["base_models"]
    meta_model  = artefacts["meta_model"]

    n_classes = 3
    stacked   = np.zeros((len(X), n_classes * len(base_models)))

    for m_idx, (name, model) in enumerate(base_models):
        probs = model.predict_proba(X)
        stacked[:, m_idx * n_classes : (m_idx + 1) * n_classes] = probs

    pred_encoded = meta_model.predict(stacked)
    pred_proba   = meta_model.predict_proba(stacked)

    result = df.copy()
    result["Predicted_Target"]   = le.inverse_transform(pred_encoded)
    result["Prob_Dropout"]       = pred_proba[:, list(le.classes_).index("Dropout")]
    result["Prob_Enrolled"]      = pred_proba[:, list(le.classes_).index("Enrolled")]
    result["Prob_Graduate"]      = pred_proba[:, list(le.classes_).index("Graduate")]
    result["Risk_Score"]         = result["Prob_Dropout"]          # 0-1, higher = more at risk
    q60 = result["Risk_Score"].quantile(0.40)   # bottom 40 % → Low
    q75 = result["Risk_Score"].quantile(0.75)   # next 35 %  → Medium  (top 25 % → High)
    joblib.dump({"q60": q60, "q75": q75}, "models/risk_thresholds.joblib")

    result["Risk_Label"] = pd.cut(
        result["Risk_Score"],
        bins=[-0.001, q60, q75, 1.001],
        labels=["Low", "Medium", "High"],
    )
    result["Engagement_Flag"]    = (
        (result["Predicted_Target"] == "Enrolled") &
        (result["Curricular units 2nd sem (approved)"] == 0)
    ).map({True: "Low Engagement", False: "Normal"})

    return result

def load_artefacts():
    if not all(os.path.exists(p) for p in
               [MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_COLS_PATH, THRESHOLDS_PATH]):
        return None, None, None, None          # <-- now returns 4 values
    artefacts    = joblib.load(MODEL_PATH)
    le           = joblib.load(LABEL_ENCODER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    thresholds   = joblib.load(THRESHOLDS_PATH)
    return artefacts, le, feature_cols, thresholds


def load_scores():
    if os.path.exists(SCORES_PATH):
        return pd.read_csv(SCORES_PATH)
    return None
