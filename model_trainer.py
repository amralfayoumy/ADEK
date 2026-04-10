"""
model_trainer.py
Trains a configurable student-outcome model stack and persists artefacts to disk.
"""

import os
import re
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
MODEL_PATH = "models/stacking_ensemble.joblib"
SCORES_PATH = "models/all_student_scores.csv"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"
THRESHOLDS_PATH = "models/risk_thresholds.joblib"


# ── Hyperparameters from notebook (Optuna-tuned defaults) ───────────────────

XGB_PARAMS = dict(
    n_estimators=929,
    alpha=2.287466581490129e-05,
    subsample=0.8766675651018592,
    colsample_bytree=0.288332829334817,
    max_depth=8,
    min_child_weight=6,
    learning_rate=0.024083411832750343,
    gamma=0.001816649055813574,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
)

LGBM_PARAMS = dict(
    n_estimators=1894,
    max_depth=23,
    learning_rate=0.024309983270196903,
    min_data_in_leaf=27,
    subsample=0.40065361124232945,
    max_bin=267,
    feature_fraction=0.1326832138080814,
    random_state=RANDOM_STATE,
    verbose=-1,
)

CATB_PARAMS = dict(
    colsample_bylevel=0.6383474716497279,
    learning_rate=0.09475494290429642,
    random_strength=0.07771221926568195,
    max_bin=490,
    depth=5,
    l2_leaf_reg=5,
    boosting_type="Plain",
    bootstrap_type="Bernoulli",
    subsample=0.8429457747642737,
    random_state=RANDOM_STATE,
    logging_level="Silent",
)

META_PARAMS = dict(
    n_estimators=47,
    alpha=6.422755620546236e-05,
    subsample=0.8452333586225941,
    colsample_bytree=0.7651776055349394,
    max_depth=3,
    min_child_weight=8,
    learning_rate=0.011014344390319484,
    gamma=6.1495867050966066e-06,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
)

MODEL_FACTORIES = {
    "xgb": XGBClassifier,
    "lgbm": LGBMClassifier,
    "catb": CatBoostClassifier,
}
MODEL_DEFAULT_PARAMS = {
    "xgb": XGB_PARAMS,
    "lgbm": LGBM_PARAMS,
    "catb": CATB_PARAMS,
}
MODEL_ORDER = ["xgb", "lgbm", "catb"]


LEAKAGE_PREFIXES_RAW = ("prob_", "predicted_", "risk_")
LEAKAGE_CANONICAL_PREFIXES = (
    "prob",
    "predicted",
    "risk",
    "pdropout",
    "ppending",
    "penrolled",
    "pgraduate",
)
LEAKAGE_EXACT = {
    "engagement_flag",
    "dropout_reason",
    "dropout reason",
    "dropoutreason",
    "target_display",
    "predicted_target_display",
    "p(dropout)",
    "p(pending)",
    "p(enrolled)",
    "p(graduate)",
}
LEAKAGE_EXACT_CANONICAL = {
    re.sub(r"[^a-z0-9]+", "", value.strip().lower()) for value in LEAKAGE_EXACT
}


# ── Core helpers ──────────────────────────────────────────────────────────────


def _canonical_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _is_leakage_feature(col_name: str) -> bool:
    raw = str(col_name).strip().lower()
    canonical = _canonical_name(col_name)

    if raw.startswith(LEAKAGE_PREFIXES_RAW):
        return True
    if canonical in LEAKAGE_EXACT_CANONICAL:
        return True
    if canonical.startswith(LEAKAGE_CANONICAL_PREFIXES):
        return True
    return False


def _build_base_models(selected_models=None, model_params=None):
    selected_models = selected_models or MODEL_ORDER
    model_params = model_params or {}

    unknown = sorted(set(selected_models) - set(MODEL_FACTORIES.keys()))
    if unknown:
        raise ValueError(f"Unknown model(s) requested: {unknown}")

    ordered = [name for name in MODEL_ORDER if name in set(selected_models)]
    if not ordered:
        raise ValueError("At least one base model must be selected for training.")

    built = []
    for name in ordered:
        params = deepcopy(MODEL_DEFAULT_PARAMS[name])
        overrides = model_params.get(name, {})
        if overrides:
            if not isinstance(overrides, dict):
                raise ValueError(f"Hyperparameters for {name} must be a dictionary.")
            params.update(overrides)
        built.append((name, MODEL_FACTORIES[name](**params)))
    return built


def _infer_preprocess_kind(series: pd.Series):
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    sample = series.dropna()
    if sample.empty:
        return "categorical_code"

    nunique = int(sample.nunique(dropna=True))
    ratio = nunique / max(len(sample), 1)
    if nunique <= 40 or ratio <= 0.08:
        return "categorical_code"
    return "categorical_frequency"


def _build_preprocess_info(df: pd.DataFrame, feature_cols: list):
    info = {}

    for col in feature_cols:
        s = df[col] if col in df.columns else pd.Series(index=df.index, dtype="float64")
        kind = _infer_preprocess_kind(s)

        if kind == "numeric":
            numeric = pd.to_numeric(s, errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            info[col] = {"kind": "numeric", "fill": fill}
        elif kind == "bool":
            info[col] = {"kind": "bool"}
        elif kind == "categorical_code":
            text = s.astype("string").fillna("__MISSING__").astype(str)
            categories = sorted(text.unique().tolist())
            mapping = {val: float(idx) for idx, val in enumerate(categories)}
            info[col] = {"kind": "categorical_code", "mapping": mapping, "unknown": -1.0}
        else:
            text = s.astype("string").fillna("__MISSING__").astype(str)
            freq = text.value_counts(normalize=True).to_dict()
            info[col] = {"kind": "categorical_frequency", "mapping": freq}

    return info


def _build_model_matrix(df: pd.DataFrame, feature_cols: list, preprocess_info=None):
    """
    Builds an inference/training matrix in the exact feature order expected
    by persisted models, with automatic preprocessing by type/distribution.
    """
    preprocess_info = preprocess_info or {}
    X = pd.DataFrame(index=df.index)
    missing_cols = []

    for col in feature_cols:
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series(np.nan, index=df.index)
            missing_cols.append(col)

        cfg = preprocess_info.get(col)
        kind = (cfg or {}).get("kind", "numeric")

        if kind == "numeric":
            fill = float((cfg or {}).get("fill", 0.0))
            X[col] = pd.to_numeric(s, errors="coerce").fillna(fill)
        elif kind == "bool":
            X[col] = s.astype("boolean").fillna(False).astype(int)
        elif kind == "categorical_code":
            mapping = (cfg or {}).get("mapping", {})
            unknown = float((cfg or {}).get("unknown", -1.0))
            text = s.astype("string").fillna("__MISSING__").astype(str)
            X[col] = text.map(mapping).fillna(unknown).astype(float)
        elif kind == "categorical_frequency":
            mapping = (cfg or {}).get("mapping", {})
            text = s.astype("string").fillna("__MISSING__").astype(str)
            X[col] = text.map(mapping).fillna(0.0).astype(float)
        else:
            X[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)

    if missing_cols:
        print(f"Warning: {len(missing_cols)} missing feature(s) backfilled: {missing_cols}")
    return X


def _select_trainable_features(df: pd.DataFrame, selected_features=None):
    """
    Selects feature columns and builds a preprocessed model matrix.
    Leakage/output columns are always excluded even if requested.
    """
    all_feature_cols = [c for c in df.columns if c != "Target"]

    if selected_features:
        selected = [c for c in selected_features if c in all_feature_cols]
        unknown = [c for c in selected_features if c not in all_feature_cols]
        if unknown:
            print("Warning: requested features not found and ignored:", unknown)
        candidate_pool = selected
    else:
        candidate_pool = all_feature_cols

    leakage_cols = [c for c in candidate_pool if _is_leakage_feature(c)]
    candidate_cols = [c for c in candidate_pool if c not in leakage_cols]

    if leakage_cols:
        print("Ignoring leakage/output columns:", leakage_cols)

    if not candidate_cols:
        raise ValueError("No trainable features available after leakage filtering.")

    preprocess_info = _build_preprocess_info(df, candidate_cols)
    X = _build_model_matrix(df, candidate_cols, preprocess_info=preprocess_info)
    return X, candidate_cols, leakage_cols, preprocess_info


def _fit_base_models_cv(X: pd.DataFrame, y: pd.Series, cv, base_models):
    """
    Returns OOF probability matrix and list of fully retrained base models.
    """
    n_classes = 3
    n_models = len(base_models)
    oof_probs = np.zeros((len(X), n_classes * n_models))
    fitted = []

    for m_idx, (name, model) in enumerate(base_models):
        print(f"  Training {name} ...")
        fold_probs = np.zeros((len(X), n_classes))

        for _fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
            Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
            Xval = X.iloc[val_idx]
            clone = model.__class__(**model.get_params())
            clone.fit(Xtr, ytr)
            fold_probs[val_idx] = clone.predict_proba(Xval)

        full_model = model.__class__(**model.get_params())
        full_model.fit(X, y)
        fitted.append((name, full_model))

        col_start = m_idx * n_classes
        oof_probs[:, col_start : col_start + n_classes] = fold_probs
        acc = accuracy_score(y, np.argmax(fold_probs, axis=1))
        print(f"    OOF accuracy: {acc:.4f}")

    return oof_probs, fitted


def train_and_save(
    data_path: str = "data.csv",
    selected_features=None,
    selected_models=None,
    model_params=None,
):
    """
    Full training pipeline with configurable feature/model/hyperparameter choices.
    Saves artefacts under models/ and returns the scored DataFrame.
    """
    os.makedirs("models", exist_ok=True)

    print("Loading data …")
    df = pd.read_csv(data_path)
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, sep=";")

    df.columns = df.columns.str.strip()

    if "Target" not in df.columns:
        raise ValueError("'Target' column not found in the CSV.")

    X, feature_cols, blocked_cols, preprocess_info = _select_trainable_features(
        df, selected_features=selected_features
    )
    y_raw = df["Target"].copy()

    if blocked_cols:
        print(f"Blocked leakage feature(s): {blocked_cols}")
    print(f"Using {len(feature_cols)} trainable feature(s).")

    base_models = _build_base_models(selected_models=selected_models, model_params=model_params)
    print("Selected base model(s):", [name for name, _ in base_models])

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), name="Target")
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\nTraining base models (5-fold CV) …")
    oof_probs, fitted_base = _fit_base_models_cv(X, y, cv, base_models)

    if len(fitted_base) >= 2:
        print("\nTraining meta-learner …")
        meta_model = XGBClassifier(**META_PARAMS)
        meta_model.fit(oof_probs, y)
        meta_preds = meta_model.predict(oof_probs)
        print(f"  Meta-model accuracy (train): {accuracy_score(y, meta_preds):.4f}")
        ensemble_mode = "stacking"
    else:
        meta_model = None
        single_preds = fitted_base[0][1].predict(X)
        print(f"\nSingle-model accuracy (train): {accuracy_score(y, single_preds):.4f}")
        ensemble_mode = "single"

    artefacts = {
        "base_models": fitted_base,
        "meta_model": meta_model,
        "ensemble_mode": ensemble_mode,
        "selected_models": [name for name, _ in fitted_base],
        "preprocess_info": preprocess_info,
    }

    joblib.dump(artefacts, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    print(f"\nModels saved to {MODEL_PATH}")

    scored_df = score_dataset(df, artefacts, le, feature_cols, preprocess_info=preprocess_info)
    scored_df.to_csv(SCORES_PATH, index=False)
    print(f"Scores saved to {SCORES_PATH}")

    return scored_df


def score_dataset(df: pd.DataFrame, artefacts: dict, le, feature_cols: list, preprocess_info=None):
    """Score every row and attach predictions + probabilities."""
    preprocess_info = preprocess_info or artefacts.get("preprocess_info", {})
    X = _build_model_matrix(df, feature_cols, preprocess_info=preprocess_info)

    base_models = artefacts["base_models"]
    meta_model = artefacts.get("meta_model")
    ensemble_mode = artefacts.get("ensemble_mode", "stacking")

    if ensemble_mode == "stacking" and meta_model is not None and len(base_models) >= 2:
        n_classes = 3
        stacked = np.zeros((len(X), n_classes * len(base_models)))

        for m_idx, (_name, model) in enumerate(base_models):
            probs = model.predict_proba(X)
            stacked[:, m_idx * n_classes : (m_idx + 1) * n_classes] = probs

        pred_encoded = meta_model.predict(stacked)
        pred_proba = meta_model.predict_proba(stacked)
    else:
        _name, model = base_models[0]
        pred_encoded = model.predict(X)
        pred_proba = model.predict_proba(X)

    result = df.copy()
    result["Predicted_Target"] = le.inverse_transform(pred_encoded)
    result["Prob_Dropout"] = pred_proba[:, list(le.classes_).index("Dropout")]
    result["Prob_Enrolled"] = pred_proba[:, list(le.classes_).index("Enrolled")]
    result["Prob_Graduate"] = pred_proba[:, list(le.classes_).index("Graduate")]
    result["Risk_Score"] = result["Prob_Dropout"]

    q60 = result["Risk_Score"].quantile(0.40)
    q75 = result["Risk_Score"].quantile(0.75)
    joblib.dump({"q60": q60, "q75": q75}, THRESHOLDS_PATH)

    result["Risk_Label"] = pd.cut(
        result["Risk_Score"],
        bins=[-0.001, q60, q75, 1.001],
        labels=["Low", "Medium", "High"],
    )
    result["Engagement_Flag"] = (
        (result["Predicted_Target"] == "Enrolled")
        & (result["Curricular units 2nd sem (approved)"] == 0)
    ).map({True: "Low Engagement", False: "Normal"})

    return result


def prepare_feature_matrix(df: pd.DataFrame, artefacts: dict, feature_cols: list):
    preprocess_info = artefacts.get("preprocess_info", {}) if isinstance(artefacts, dict) else {}
    return _build_model_matrix(df, feature_cols, preprocess_info=preprocess_info)


def load_artefacts():
    if not all(
        os.path.exists(p)
        for p in [MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_COLS_PATH, THRESHOLDS_PATH]
    ):
        return None, None, None, None
    artefacts = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    thresholds = joblib.load(THRESHOLDS_PATH)
    return artefacts, le, feature_cols, thresholds


def load_scores():
    if os.path.exists(SCORES_PATH):
        return pd.read_csv(SCORES_PATH)
    return None
