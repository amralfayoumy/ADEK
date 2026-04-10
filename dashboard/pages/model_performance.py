import hashlib
import io
import json
import os
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dashboard.constants import OUTCOME_DISPLAY_ORDER, OUTCOME_LABEL_MAP, OUTCOME_RAW_ORDER
from dashboard.ui import dark_layout


DATA_PATH = "data.csv"
REGISTRY_PATH = "models/model_registry.json"
CANDIDATE_ROOT = "models/candidates"
VERSIONS_ROOT = "models/versions"
PENDING_CANDIDATE_KEY = "mlops_pending_candidate"
SECTION_SUMMARY = ":material/fact_check: MLOps Summary"
SECTION_EVAL = ":material/analytics: Evaluation"
SECTION_DATA = ":material/database: Data & Drift"
SECTION_VERSIONING = ":material/deployed_code: Versioning"
SECTION_RETRAIN = ":material/model_training: Retrain"
ACTIVE_SECTION_KEY = "mlops_active_section"
PENDING_SECTION_KEY = "mlops_pending_active_section"
ETL_REPORT_KEY = "mlops_last_etl_report"
ETL_LOGS_KEY = "mlops_last_etl_logs"
RETRAIN_LOGS_KEY = "mlops_last_retrain_logs"
LOG_LEVEL_COLORS = {
    "INFO": "#1d4ed8",
    "WARN": "#b45309",
    "ERROR": "#b91c1c",
}
SECTION_OPTIONS = [
    SECTION_SUMMARY,
    SECTION_EVAL,
    SECTION_DATA,
    SECTION_VERSIONING,
    SECTION_RETRAIN,
]


def _queue_active_section(section):
    if section in SECTION_OPTIONS:
        st.session_state[PENDING_SECTION_KEY] = section


def _parse_log_line(line):
    text = str(line).strip()
    if not text:
        return None

    match = re.match(r"^\[(INFO|WARN|ERROR)\]\s*(.*)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).strip()

    lowered = text.lower()
    if any(token in lowered for token in ["error", "failed", "exception", "traceback"]):
        return "ERROR", text
    if "warning" in lowered or "warn" in lowered:
        return "WARN", text
    return "INFO", text


def _render_color_logs(logs, max_lines=300):
    if logs is None:
        st.caption("No logs available.")
        return

    if isinstance(logs, str):
        lines = logs.splitlines()
    elif isinstance(logs, (list, tuple)):
        lines = [str(item) for item in logs]
    else:
        lines = [str(logs)]

    lines = [line for line in lines if str(line).strip()]
    if not lines:
        st.caption("No logs available.")
        return

    lines = lines[-max_lines:]
    html_lines = []
    for line in lines:
        parsed = _parse_log_line(line)
        if parsed is None:
            continue
        level, message = parsed
        color = LOG_LEVEL_COLORS.get(level, LOG_LEVEL_COLORS["INFO"])
        html_lines.append(
            "<div style='font-family:Consolas,\"Courier New\",monospace;"
            "font-size:0.82rem;line-height:1.45;'>"
            f"<span style='color:{color};font-weight:700;'>[{level}]</span> "
            f"<span>{escape(message)}</span>"
            "</div>"
        )

    st.markdown("".join(html_lines), unsafe_allow_html=True)


def _simulate_etl_refresh(data_path=DATA_PATH):
    logs = []

    def log(msg, level="INFO"):
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        logs.append(f"[{level.upper()}] [{stamp}] {msg}")

    log("Starting orchestrated ETL refresh.")
    log("Extract complete: Student Information System feed.")
    log("Extract complete: Academic records feed.")
    log("Extract complete: Finance and attendance feed.")
    log("Applying schema harmonization and entity resolution rules.")

    df = _safe_read_csv(data_path)
    if df is None:
        raise FileNotFoundError(
            f"ETL refresh failed: could not read {data_path}."
        )

    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    dataset_sha = _sha256_of_file(data_path)[:16]
    dataset_fp = _dataset_fingerprint(df)[:16]

    log(
        f"Curated dataset assembled with {len(df):,} rows and {df.shape[1]} columns."
    )
    if duplicate_rows > 0 or missing_cells > 0:
        log(
            "Data quality checks completed with warnings "
            f"(duplicates={duplicate_rows:,}, missing cells={missing_cells:,}).",
            level="WARN",
        )
    else:
        log(
            "Data quality checks passed "
            f"(duplicates={duplicate_rows:,}, missing cells={missing_cells:,})."
        )
    log("Published refreshed analytics-ready dataset snapshot.")

    report = {
        "refreshed_utc": _utc_now_iso(),
        "dataset_path": data_path,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": duplicate_rows,
        "missing_cells": missing_cells,
        "dataset_sha256": dataset_sha,
        "dataset_fingerprint": dataset_fp,
    }
    return report, logs


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        try:
            df = pd.read_csv(path, sep=";")
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            return None


def _sha256_of_file(path, chunk_size=1024 * 1024):
    if not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_paths(mt):
    return [
        getattr(mt, "MODEL_PATH", "models/stacking_ensemble.joblib"),
        getattr(mt, "LABEL_ENCODER_PATH", "models/label_encoder.joblib"),
        getattr(mt, "FEATURE_COLS_PATH", "models/feature_cols.joblib"),
        getattr(mt, "THRESHOLDS_PATH", "models/risk_thresholds.joblib"),
        getattr(mt, "SCORES_PATH", "models/all_student_scores.csv"),
    ]


def _file_descriptor(path):
    exists = os.path.exists(path)
    if not exists:
        return {
            "Path": path,
            "Exists": False,
            "Size (MB)": 0.0,
            "Modified (UTC)": "-",
            "SHA256": "-",
        }

    stat = os.stat(path)
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    return {
        "Path": path,
        "Exists": True,
        "Size (MB)": round(stat.st_size / (1024 * 1024), 4),
        "Modified (UTC)": modified,
        "SHA256": _sha256_of_file(path)[:16],
    }


def _load_registry(path=REGISTRY_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _save_registry(entries, path=REGISTRY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _normalize_version(version):
    if version is None:
        return None
    text = str(version).strip()
    if text.startswith("v"):
        text = text[1:]
    parts = text.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    major, minor, patch = map(int, parts)
    return f"v{major}.{minor}.{patch}"


def _version_tuple(version):
    normalized = _normalize_version(version)
    if not normalized:
        return None
    major, minor, patch = normalized[1:].split(".")
    return int(major), int(minor), int(patch)


def _next_version(entries):
    max_tuple = None
    for entry in entries:
        vt = _version_tuple(entry.get("version"))
        if vt and (max_tuple is None or vt > max_tuple):
            max_tuple = vt
    if max_tuple is None:
        return "v1.0.0"
    major, minor, patch = max_tuple
    return f"v{major}.{minor}.{patch + 1}"


def _dataset_fingerprint(df):
    if df is None or df.empty:
        return ""

    sample = df.head(min(1500, len(df)))
    header_blob = "|".join([str(df.shape[0]), str(df.shape[1]), *map(str, df.columns)])
    row_hashes = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
    digest = hashlib.sha256(header_blob.encode("utf-8") + row_hashes).hexdigest()
    return digest


def _metrics_from_predictions(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro.get("precision", 0.0),
        "macro_recall": macro.get("recall", 0.0),
        "macro_f1": macro.get("f1-score", 0.0),
        "weighted_f1": weighted.get("f1-score", 0.0),
    }


def _population_stability_index(expected, actual, bins=10):
    if len(expected) < 50 or len(actual) < 50:
        return np.nan

    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return np.nan

    exp_hist, _ = np.histogram(expected, bins=cuts)
    act_hist, _ = np.histogram(actual, bins=cuts)

    exp_pct = exp_hist / max(exp_hist.sum(), 1)
    act_pct = act_hist / max(act_hist.sum(), 1)

    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _drift_watchlist(df):
    if "Enrollment_Year" not in df.columns:
        return pd.DataFrame()

    years = sorted(pd.Series(df["Enrollment_Year"]).dropna().unique().tolist())
    if len(years) < 2:
        return pd.DataFrame()

    baseline_year, current_year = years[0], years[-1]
    baseline = df[df["Enrollment_Year"] == baseline_year]
    current = df[df["Enrollment_Year"] == current_year]

    numeric_candidates = [
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    ]
    numeric_cols = [
        c
        for c in numeric_candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        exp = pd.to_numeric(baseline[col], errors="coerce").dropna()
        act = pd.to_numeric(current[col], errors="coerce").dropna()
        psi = _population_stability_index(exp.values, act.values)
        if np.isnan(psi):
            status = "Insufficient Data"
        elif psi < 0.10:
            status = "Stable"
        elif psi < 0.20:
            status = "Watch"
        else:
            status = "Drift"
        rows.append(
            {
                "Feature": col,
                "Baseline Year": baseline_year,
                "Current Year": current_year,
                "PSI": None if np.isnan(psi) else round(psi, 4),
                "Status": status,
            }
        )

    out = pd.DataFrame(rows)
    out["PSI"] = pd.to_numeric(out["PSI"], errors="coerce")
    return out.sort_values("PSI", ascending=False, na_position="last")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _is_blacklisted_feature(col_name, mt):
    checker = getattr(mt, "_is_leakage_feature", None)
    if callable(checker):
        try:
            return bool(checker(col_name))
        except Exception:
            pass

    raw = str(col_name).strip().lower()
    compact = "".join(ch for ch in raw if ch.isalnum())
    if raw.startswith(("prob_", "predicted_", "risk_")):
        return True
    if compact.startswith(("prob", "predicted", "risk", "pdropout", "ppending", "penrolled", "pgraduate")):
        return True
    if compact in {"engagementflag", "dropoutreason", "targetdisplay", "predictedtargetdisplay"}:
        return True
    return False


def _trainer_path_map(mt):
    return {
        "model_path": getattr(mt, "MODEL_PATH", "models/stacking_ensemble.joblib"),
        "scores_path": getattr(mt, "SCORES_PATH", "models/all_student_scores.csv"),
        "label_encoder_path": getattr(mt, "LABEL_ENCODER_PATH", "models/label_encoder.joblib"),
        "feature_cols_path": getattr(mt, "FEATURE_COLS_PATH", "models/feature_cols.joblib"),
        "thresholds_path": getattr(mt, "THRESHOLDS_PATH", "models/risk_thresholds.joblib"),
    }


def _required_artifact_keys():
    return [
        "model_path",
        "scores_path",
        "label_encoder_path",
        "feature_cols_path",
        "thresholds_path",
    ]


def _version_path_map(version):
    normalized = _normalize_version(version)
    if not normalized:
        raise ValueError(f"Invalid version format: {version}")
    base_dir = os.path.join(VERSIONS_ROOT, normalized)
    return {
        "base_dir": base_dir,
        "model_path": os.path.join(base_dir, "stacking_ensemble.joblib"),
        "scores_path": os.path.join(base_dir, "all_student_scores.csv"),
        "label_encoder_path": os.path.join(base_dir, "label_encoder.joblib"),
        "feature_cols_path": os.path.join(base_dir, "feature_cols.joblib"),
        "thresholds_path": os.path.join(base_dir, "risk_thresholds.joblib"),
    }


def _candidate_path_map(candidate_id):
    base_dir = os.path.join(CANDIDATE_ROOT, candidate_id)
    return {
        "base_dir": base_dir,
        "model_path": os.path.join(base_dir, "stacking_ensemble.joblib"),
        "scores_path": os.path.join(base_dir, "all_student_scores.csv"),
        "label_encoder_path": os.path.join(base_dir, "label_encoder.joblib"),
        "feature_cols_path": os.path.join(base_dir, "feature_cols.joblib"),
        "thresholds_path": os.path.join(base_dir, "risk_thresholds.joblib"),
    }


def _apply_trainer_path_map(mt, path_map):
    mt.MODEL_PATH = path_map["model_path"]
    mt.SCORES_PATH = path_map["scores_path"]
    mt.LABEL_ENCODER_PATH = path_map["label_encoder_path"]
    mt.FEATURE_COLS_PATH = path_map["feature_cols_path"]
    mt.THRESHOLDS_PATH = path_map["thresholds_path"]


def _copy_artifact_bundle(src_paths, dst_paths):
    missing = [
        key
        for key in _required_artifact_keys()
        if not os.path.exists(src_paths.get(key, ""))
    ]
    if missing:
        raise FileNotFoundError(f"Missing artefacts for copy: {missing}")

    for key in _required_artifact_keys():
        src = src_paths[key]
        dst = dst_paths[key]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _snapshot_production_version(mt, version):
    production_paths = _trainer_path_map(mt)
    version_paths = _version_path_map(version)
    os.makedirs(version_paths["base_dir"], exist_ok=True)
    _copy_artifact_bundle(production_paths, version_paths)
    return version_paths


def _snapshot_exists_for_version(version):
    try:
        version_paths = _version_path_map(version)
    except Exception:
        return False
    return all(os.path.exists(version_paths.get(key, "")) for key in _required_artifact_keys())


def _removed_versions(entries):
    removed = set()
    for entry in entries:
        rv = _normalize_version(entry.get("removed_version"))
        if rv:
            removed.add(rv)
    return removed


def _version_catalog_rows(entries, current_version):
    versions = []
    for entry in entries:
        ver = _normalize_version(entry.get("version"))
        if ver:
            versions.append(ver)

    unique_versions = sorted(set(versions), key=lambda v: _version_tuple(v) or (0, 0, 0))
    removed_versions = _removed_versions(entries)
    rows = []

    for ver in unique_versions:
        if ver in removed_versions:
            continue

        ver_events = [e for e in entries if _normalize_version(e.get("version")) == ver]
        latest_event = ver_events[-1] if ver_events else {}
        exists = _snapshot_exists_for_version(ver)
        if ver == current_version:
            status = "Active"
        elif not exists:
            status = "Unavailable"
        else:
            status = "Available"

        rows.append(
            {
                "Version": ver,
                "Status": status,
                "Snapshot": "Yes" if exists else "No",
                "Last Event": latest_event.get("trigger", "unknown"),
                "Last Updated": latest_event.get("timestamp_utc", "-"),
                "Accuracy": latest_event.get("accuracy"),
                "Macro F1": latest_event.get("macro_f1"),
            }
        )
    return rows


def _rollback_to_version(mt, target_version, baseline_metrics):
    target = _normalize_version(target_version)
    if not target:
        raise ValueError("Invalid rollback target version.")

    source_paths = _version_path_map(target)
    production_paths = _trainer_path_map(mt)
    _copy_artifact_bundle(source_paths, production_paths)

    restored_df = _safe_read_csv(source_paths["scores_path"])
    if restored_df is not None and {"Target", "Predicted_Target"}.issubset(restored_df.columns):
        restored_metrics = _metrics_from_predictions(
            restored_df["Target"], restored_df["Predicted_Target"]
        )
        restored_rows = int(len(restored_df))
        restored_fingerprint = _dataset_fingerprint(restored_df)[:16]
    else:
        restored_metrics = {
            "accuracy": _safe_float(baseline_metrics.get("accuracy"), 0.0),
            "macro_f1": _safe_float(baseline_metrics.get("macro_f1"), 0.0),
            "macro_recall": _safe_float(baseline_metrics.get("macro_recall"), 0.0),
        }
        restored_rows = 0
        restored_fingerprint = ""

    feature_count = _feature_count_from_joblib(source_paths["feature_cols_path"])

    entries = _load_registry()
    previous_active = _normalize_version(entries[-1].get("version")) if entries else "v0.0.0-untracked"
    entries.append(
        {
            "version": target,
            "timestamp_utc": _utc_now_iso(),
            "trigger": "manual_rollback",
            "rollback_from_version": previous_active,
            "runtime_sec": 0.0,
            "rows": restored_rows,
            "features": feature_count,
            "accuracy": round(_safe_float(restored_metrics.get("accuracy"), 0.0), 6),
            "macro_f1": round(_safe_float(restored_metrics.get("macro_f1"), 0.0), 6),
            "macro_recall": round(_safe_float(restored_metrics.get("macro_recall"), 0.0), 6),
            "dataset_path": DATA_PATH,
            "dataset_sha256": _sha256_of_file(DATA_PATH)[:16],
            "dataset_fingerprint": restored_fingerprint,
            "python_version": platform.python_version(),
        }
    )
    _save_registry(entries)
    return target, previous_active


def _remove_version_snapshot(version, current_version):
    target = _normalize_version(version)
    current = _normalize_version(current_version)

    if not target:
        raise ValueError("Invalid version selected for removal.")
    if current and target == current:
        raise ValueError("Active model version cannot be removed.")

    version_paths = _version_path_map(target)
    if os.path.isdir(version_paths["base_dir"]):
        shutil.rmtree(version_paths["base_dir"], ignore_errors=True)

    entries = _load_registry()
    anchor_version = current or target
    entries.append(
        {
            "version": anchor_version,
            "timestamp_utc": _utc_now_iso(),
            "trigger": "manual_version_delete",
            "removed_version": target,
            "runtime_sec": 0.0,
            "python_version": platform.python_version(),
        }
    )
    _save_registry(entries)


def _feature_count_from_joblib(path):
    if not os.path.exists(path):
        return None
    try:
        cols = joblib.load(path)
        return int(len(cols))
    except Exception:
        return None


def _train_candidate_model(
    mt,
    data_path=DATA_PATH,
    selected_features=None,
    selected_models=None,
    model_params=None,
):
    candidate_id = datetime.now(timezone.utc).strftime("cand_%Y%m%d_%H%M%S")
    candidate_paths = _candidate_path_map(candidate_id)
    os.makedirs(candidate_paths["base_dir"], exist_ok=True)

    production_paths = _trainer_path_map(mt)
    trainer_logs = ""
    try:
        _apply_trainer_path_map(mt, candidate_paths)
        start = time.perf_counter()
        stdout_buffer = io.StringIO()
        original_stdout = os.sys.stdout
        try:
            os.sys.stdout = stdout_buffer
            scored_df = mt.train_and_save(
                data_path,
                selected_features=selected_features,
                selected_models=selected_models,
                model_params=model_params,
            )
        finally:
            os.sys.stdout = original_stdout
        trainer_logs = stdout_buffer.getvalue()
        runtime_sec = round(time.perf_counter() - start, 2)
    finally:
        _apply_trainer_path_map(mt, production_paths)

    candidate_metrics = _metrics_from_predictions(scored_df["Target"], scored_df["Predicted_Target"])
    feature_count = _feature_count_from_joblib(candidate_paths["feature_cols_path"])

    return {
        "candidate_id": candidate_id,
        "created_utc": _utc_now_iso(),
        "runtime_sec": runtime_sec,
        "rows": int(len(scored_df)),
        "features": feature_count,
        "metrics": {k: _safe_float(v) for k, v in candidate_metrics.items()},
        "selected_features": selected_features or [],
        "selected_models": selected_models or [],
        "model_params": model_params or {},
        "dataset_path": data_path,
        "dataset_sha256": _sha256_of_file(data_path)[:16],
        "dataset_fingerprint": _dataset_fingerprint(scored_df)[:16],
        "paths": candidate_paths,
        "trainer_logs": trainer_logs,
    }


def _promote_candidate_model(mt, candidate_info, baseline_metrics):
    candidate_paths = candidate_info.get("paths", {})
    required_keys = _required_artifact_keys()
    missing = [key for key in required_keys if not os.path.exists(candidate_paths.get(key, ""))]
    if missing:
        raise FileNotFoundError(f"Candidate artefacts missing: {missing}")

    production_paths = _trainer_path_map(mt)
    entries = _load_registry()
    version = _next_version(entries)
    version_paths = _version_path_map(version)

    _copy_artifact_bundle(candidate_paths, production_paths)
    os.makedirs(version_paths["base_dir"], exist_ok=True)
    _copy_artifact_bundle(candidate_paths, version_paths)

    cand_metrics = candidate_info.get("metrics", {})

    entries.append(
        {
            "version": version,
            "timestamp_utc": _utc_now_iso(),
            "trigger": "manual_candidate_promotion",
            "candidate_id": candidate_info.get("candidate_id"),
            "snapshot_dir": version_paths["base_dir"],
            "runtime_sec": _safe_float(candidate_info.get("runtime_sec"), 0.0),
            "rows": int(candidate_info.get("rows", 0)),
            "features": candidate_info.get("features"),
            "accuracy": round(_safe_float(cand_metrics.get("accuracy"), 0.0), 6),
            "macro_f1": round(_safe_float(cand_metrics.get("macro_f1"), 0.0), 6),
            "macro_recall": round(_safe_float(cand_metrics.get("macro_recall"), 0.0), 6),
            "baseline_accuracy": round(_safe_float(baseline_metrics.get("accuracy"), 0.0), 6),
            "baseline_macro_f1": round(_safe_float(baseline_metrics.get("macro_f1"), 0.0), 6),
            "dataset_path": candidate_info.get("dataset_path", DATA_PATH),
            "dataset_sha256": candidate_info.get("dataset_sha256", ""),
            "dataset_fingerprint": candidate_info.get("dataset_fingerprint", ""),
            "python_version": platform.python_version(),
        }
    )
    _save_registry(entries)
    return version


def _discard_candidate_model(candidate_info):
    candidate_dir = candidate_info.get("paths", {}).get("base_dir")
    if candidate_dir and os.path.isdir(candidate_dir):
        shutil.rmtree(candidate_dir, ignore_errors=True)


def render(df_full, get_trainer, display_outcome):
    st.markdown("# :material/monitoring: Model Management & MLOps")
    st.caption("Centralized monitoring, governance, versioning, and retraining controls.")

    required_cols = {"Target", "Predicted_Target"}
    if not required_cols.issubset(df_full.columns):
        st.error("Model evaluation columns are missing in the scored dataset.")
        return

    mt = get_trainer()
    y_true = df_full["Target"]
    y_pred = df_full["Predicted_Target"]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().rename(index=OUTCOME_LABEL_MAP)
    metrics = _metrics_from_predictions(y_true, y_pred)
    art, le_obj, feat_cols, thresholds = mt.load_artefacts()

    source_df = _safe_read_csv(DATA_PATH)
    source_rows = int(len(source_df)) if source_df is not None else int(len(df_full))
    source_cols = int(source_df.shape[1]) if source_df is not None else int(df_full.shape[1])
    source_missing_pct = (
        float(source_df.isna().sum().sum()) / max(source_df.size, 1)
        if source_df is not None
        else float(df_full.isna().sum().sum()) / max(df_full.size, 1)
    )

    registry = _load_registry()
    current_version = (
        _normalize_version(registry[-1].get("version")) if registry else None
    ) or "v0.0.0-untracked"
    last_retrain = registry[-1].get("timestamp_utc") if registry else "Not tracked"

    alert = st.session_state.pop("mlops_retrain_notice", None)
    if alert:
        st.success(alert)

    pending_candidate = st.session_state.get(PENDING_CANDIDATE_KEY)
    if pending_candidate:
        pending_scores = pending_candidate.get("paths", {}).get("scores_path", "")
        if not pending_scores or not os.path.exists(pending_scores):
            st.session_state.pop(PENDING_CANDIDATE_KEY, None)
            pending_candidate = None
            st.warning(
                "A pending candidate review was cleared because its artefacts were missing."
            )

    if pending_candidate:
        st.warning(
            f"Pending candidate {pending_candidate.get('candidate_id', 'unknown')} is waiting for review and approval. "
            "Production remains unchanged until promotion is approved."
        )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Model Version", current_version)
    c2.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c3.metric("Macro F1", f"{metrics['macro_f1'] * 100:.2f}%")
    c4.metric("Dataset Rows", f"{source_rows:,}")
    c5.metric("Dataset Features", max(source_cols - 1, 0))
    c6.metric("Missing Cells", f"{source_missing_pct * 100:.2f}%")

    st.caption(f"Last model event: {last_retrain}")

    pending_section = st.session_state.pop(PENDING_SECTION_KEY, None)
    if pending_section in SECTION_OPTIONS:
        st.session_state[ACTIVE_SECTION_KEY] = pending_section
    if ACTIVE_SECTION_KEY not in st.session_state:
        st.session_state[ACTIVE_SECTION_KEY] = SECTION_SUMMARY

    active_section = st.radio(
        "MLOps Section",
        options=SECTION_OPTIONS,
        horizontal=True,
        key=ACTIVE_SECTION_KEY,
    )

    ensemble_mode = art.get("ensemble_mode", "stacking") if art else "unknown"
    active_model_names = [name for name, _ in art.get("base_models", [])] if art else []
    model_family = (
        "Stacking ensemble"
        if ensemble_mode == "stacking" and len(active_model_names) >= 2
        else (
            f"Single-model ({active_model_names[0].upper()})"
            if active_model_names
            else "Unavailable"
        )
    )

    if active_section == SECTION_SUMMARY:
        st.markdown("#### Model Card")

        left, right = st.columns(2)
        with left:
            base_models_text = ", ".join([m.upper() for m in active_model_names]) if active_model_names else "Unavailable"
            meta_text = "XGBoost meta-learner" if ensemble_mode == "stacking" and len(active_model_names) >= 2 else "Not used (single-model mode)"
            st.markdown(
                "\n".join(
                    [
                        f"- **Model Family:** {model_family}",
                        f"- **Base Models:** {base_models_text}",
                        f"- **Meta Learner:** {meta_text}",
                        "- **Prediction Classes:** Dropout / Pending / Graduate",
                        "- **Serving Mode:** Batch scoring persisted to models/all_student_scores.csv",
                    ]
                )
            )
        with right:
            classes = [display_outcome(c) for c in le_obj.classes_] if le_obj is not None else []
            st.markdown(
                "\n".join(
                    [
                        f"- **Tracked Version:** {current_version}",
                        f"- **Feature Count:** {len(feat_cols) if feat_cols is not None else 'Unknown'}",
                        f"- **Runtime Environment:** Python {platform.python_version()}",
                        f"- **Class Labels in Encoder:** {', '.join(classes) if classes else 'Unavailable'}",
                        "- **Risk Thresholding:** Quantile based (`q60`, `q75`)",
                    ]
                )
            )

        check_rows = [
            {
                "Check": "Training data source available (data.csv)",
                "Status": "Pass" if os.path.exists(DATA_PATH) else "Fail",
            },
            {
                "Check": "Core artefacts loaded",
                "Status": "Pass" if art is not None else "Fail",
            },
            {
                "Check": "Risk thresholds available",
                "Status": "Pass" if thresholds is not None else "Fail",
            },
            {
                "Check": "Model registry tracking enabled",
                "Status": "Pass" if len(registry) > 0 else "Warning",
            },
            {
                "Check": "All expected classes present in predictions",
                "Status": (
                    "Pass"
                    if set(OUTCOME_RAW_ORDER).issubset(set(df_full["Predicted_Target"].unique()))
                    else "Warning"
                ),
            },
        ]
        st.markdown("#### Operational Readiness")
        st.dataframe(pd.DataFrame(check_rows), width="stretch")

        st.info(
            "Pending is the dashboard display name for the model's internal Enrolled class. "
            "This page evaluates historical outcomes for validation and MLOps governance."
        )

    if active_section == SECTION_EVAL:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Classification Report")
            st.dataframe(report_df.style.background_gradient(cmap="Blues"), width="stretch")
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
            st.plotly_chart(fig_cm, width="stretch")

        class_rows = []
        for raw_label in OUTCOME_RAW_ORDER:
            if raw_label in report:
                class_rows.append(
                    {
                        "Class": display_outcome(raw_label),
                        "Precision": report[raw_label].get("precision", 0.0),
                        "Recall": report[raw_label].get("recall", 0.0),
                        "F1": report[raw_label].get("f1-score", 0.0),
                    }
                )
        if class_rows:
            cls_df = pd.DataFrame(class_rows)
            cls_melt = cls_df.melt(id_vars="Class", var_name="Metric", value_name="Score")
            fig_class = px.bar(
                cls_melt,
                x="Class",
                y="Score",
                color="Metric",
                barmode="group",
                text_auto=".2f",
                title="Per-Class Precision/Recall/F1",
            )
            dark_layout(fig_class)
            fig_class.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_class, width="stretch")

        st.markdown("#### Importance Diagnostics")
        if art is None:
            st.warning("Model artefacts are unavailable for feature-importance diagnostics.")
        elif ensemble_mode == "stacking" and art.get("meta_model") is not None and len(active_model_names) >= 2:
            st.caption(
                "Meta-learner importance below is based on stacking input signals "
                "(base-model class probabilities), not raw dataset columns."
            )
            meta = art["meta_model"]
            labels = []
            for bname, _ in art["base_models"]:
                for cls in le_obj.classes_:
                    labels.append(f"{bname.upper()} probability for {display_outcome(cls)}")
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
                    title="Meta-Learner Input Signal Importance (Top 15)",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                dark_layout(fig_imp, height=460)
                st.plotly_chart(fig_imp, width="stretch")
        else:
            bname, bmodel = art["base_models"][0]
            if hasattr(bmodel, "feature_importances_") and feat_cols is not None:
                importances = np.asarray(bmodel.feature_importances_)
                n = min(len(importances), len(feat_cols))
                if n == 0:
                    st.info("Feature importances are unavailable for the selected single model.")
                else:
                    imp_df = (
                        pd.DataFrame({"Feature": feat_cols[:n], "Importance": importances[:n]})
                        .sort_values("Importance", ascending=True)
                        .tail(20)
                    )
                    fig_single = px.bar(
                        imp_df,
                        y="Feature",
                        x="Importance",
                        orientation="h",
                        title=f"{bname.upper()} Feature Importance (Top 20)",
                        color="Importance",
                        color_continuous_scale="Blues",
                    )
                    dark_layout(fig_single, height=500)
                    st.plotly_chart(fig_single, width="stretch")
            else:
                st.info("This model does not expose feature importances.")

        st.markdown("#### Base-Model Raw Feature Importance")
        if art is None or feat_cols is None:
            st.info("Base-model feature importances are unavailable because artefacts are missing.")
        else:
            per_model_importance = []
            for bname, bmodel in art.get("base_models", []):
                if not hasattr(bmodel, "feature_importances_"):
                    continue
                importances = np.asarray(bmodel.feature_importances_)
                n = min(len(importances), len(feat_cols))
                if n == 0:
                    continue

                abs_importances = np.abs(importances[:n].astype(float))
                max_abs = float(np.max(abs_importances)) if len(abs_importances) else 0.0
                if max_abs > 0:
                    importance_01 = abs_importances / max_abs
                else:
                    importance_01 = np.zeros_like(abs_importances)

                model_df = pd.DataFrame(
                    {
                        "Feature": feat_cols[:n],
                        "ImportanceAbs": abs_importances,
                        "Importance01": importance_01,
                        "Model": bname.upper(),
                    }
                )
                per_model_importance.append(model_df)

            if not per_model_importance:
                st.info("Current trained base models do not expose raw feature importances.")
            else:
                st.caption(
                    "The matrix below compares the same features across models. "
                    "Each model is normalized independently to 0-1 after taking absolute importance."
                )

                per_model_df = pd.concat(per_model_importance, ignore_index=True)
                summary_df = (
                    per_model_df.groupby("Feature", as_index=False)["Importance01"]
                    .mean()
                    .rename(columns={"Importance01": "MeanImportance01"})
                    .sort_values("MeanImportance01", ascending=False)
                )

                top_n = min(15, len(summary_df))
                top_features = summary_df.head(top_n)["Feature"].tolist()

                heatmap_df = (
                    per_model_df[per_model_df["Feature"].isin(top_features)]
                    .pivot_table(
                        index="Feature",
                        columns="Model",
                        values="Importance01",
                        aggfunc="max",
                    )
                    .fillna(0.0)
                )

                model_order = sorted(heatmap_df.columns.tolist())
                feature_order = [
                    f for f in summary_df["Feature"].tolist() if f in heatmap_df.index
                ]
                heatmap_df = heatmap_df.reindex(index=feature_order, columns=model_order)

                fig_per_model = px.imshow(
                    heatmap_df,
                    labels={
                        "x": "Model",
                        "y": "Feature",
                        "color": "Importance (0-1)",
                    },
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Top Features Across Models (Absolute, Per-Model Normalized 0-1)",
                )
                dark_layout(fig_per_model, height=620)
                st.plotly_chart(fig_per_model, width="stretch")

                table_df = (
                    summary_df[summary_df["Feature"].isin(feature_order)]
                    .set_index("Feature")
                    .join(heatmap_df, how="left")
                    .reset_index()
                    .rename(columns={"MeanImportance01": "Mean (All Models)"})
                )

                table_df["Mean (All Models)"] = table_df["Mean (All Models)"].round(3)
                for model_col in model_order:
                    table_df[model_col] = pd.to_numeric(
                        table_df[model_col], errors="coerce"
                    ).round(3)

                st.dataframe(table_df, width="stretch")

    if active_section == SECTION_DATA:
        st.markdown("#### Dataset Lineage")

        dataset_file = _file_descriptor(DATA_PATH)
        dataset_rows_view = [
            {
                "Artifact": "Training Data",
                "Path": dataset_file["Path"],
                "Exists": dataset_file["Exists"],
                "Size (MB)": dataset_file["Size (MB)"],
                "Modified (UTC)": dataset_file["Modified (UTC)"],
                "SHA256": dataset_file["SHA256"],
            }
        ]
        st.dataframe(pd.DataFrame(dataset_rows_view), width="stretch")

        if source_df is not None:
            total_cells = max(source_df.size, 1)
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            quality_col1.metric("Rows", f"{len(source_df):,}")
            quality_col2.metric("Columns", source_df.shape[1])
            quality_col3.metric("Duplicate Rows", int(source_df.duplicated().sum()))
            quality_col4.metric("Missing Cells", f"{(source_df.isna().sum().sum() / total_cells) * 100:.2f}%")

            if "Target" in source_df.columns:
                actual_dist = (
                    source_df["Target"]
                    .value_counts(normalize=True)
                    .rename_axis("Class")
                    .reset_index(name="Share")
                )
                actual_dist["Class"] = actual_dist["Class"].map(display_outcome).fillna(
                    actual_dist["Class"]
                )
                fig_actual = px.bar(
                    actual_dist,
                    x="Class",
                    y="Share",
                    text_auto=".1%",
                    title="Actual Target Distribution (Training Data)",
                )
                dark_layout(fig_actual)
                fig_actual.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_actual, width="stretch")

            miss_share = (
                source_df.isna().mean().sort_values(ascending=False).head(12).reset_index()
            )
            miss_share.columns = ["Feature", "MissingShare"]
            fig_missing = px.bar(
                miss_share,
                x="MissingShare",
                y="Feature",
                orientation="h",
                text_auto=".1%",
                title="Top Features by Missingness",
            )
            dark_layout(fig_missing, height=420)
            fig_missing.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_missing, width="stretch")

        st.markdown("#### Drift Watchlist (PSI)")
        drift_df = _drift_watchlist(df_full)
        if drift_df.empty:
            st.info(
                "Drift watchlist requires Enrollment_Year and enough cohort volume in at least two years."
            )
        else:
            st.dataframe(drift_df, width="stretch")

    if active_section == SECTION_VERSIONING:
        st.markdown("#### Model Artefact Inventory")
        artefact_df = pd.DataFrame(
            [_file_descriptor(path) for path in _model_artifact_paths(mt)]
            + [_file_descriptor(REGISTRY_PATH)]
        )
        st.dataframe(artefact_df, width="stretch")

        st.markdown("#### Model Registry")
        if registry:
            reg_df = pd.DataFrame(registry)
            show_cols = [
                c
                for c in [
                    "version",
                    "timestamp_utc",
                    "trigger",
                    "runtime_sec",
                    "rows",
                    "features",
                    "accuracy",
                    "macro_f1",
                    "dataset_sha256",
                ]
                if c in reg_df.columns
            ]
            st.dataframe(reg_df[show_cols], width="stretch")

            if {"version", "accuracy"}.issubset(reg_df.columns):
                removed_versions = _removed_versions(registry)
                perf_curve = reg_df[["version", "accuracy"]].copy()
                perf_curve["version"] = perf_curve["version"].map(_normalize_version)
                perf_curve = perf_curve[
                    perf_curve["version"].notna()
                    & ~perf_curve["version"].isin(removed_versions)
                ]
                perf_curve["accuracy"] = pd.to_numeric(perf_curve["accuracy"], errors="coerce")
                perf_curve = perf_curve.dropna(subset=["accuracy"])
                perf_curve = perf_curve.drop_duplicates(subset=["version"], keep="last")

                if not perf_curve.empty:
                    perf_curve["_sort"] = perf_curve["version"].map(
                        lambda v: _version_tuple(v) or (0, 0, 0)
                    )
                    perf_curve = perf_curve.sort_values("_sort").drop(columns=["_sort"])

                    fig_reg = px.line(
                        perf_curve,
                        x="version",
                        y="accuracy",
                        markers=True,
                        title="Accuracy Across Model Versions",
                    )
                    dark_layout(fig_reg)
                    fig_reg.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_reg, width="stretch")
                else:
                    st.info("No active version history with accuracy is available to plot.")

            st.markdown("#### Version Snapshots")
            version_rows = _version_catalog_rows(registry, current_version)
            version_df = pd.DataFrame(version_rows)
            st.dataframe(version_df, width="stretch")

            available_versions = [
                row["Version"] for row in version_rows if row["Status"] == "Available"
            ]
            rollback_candidates = [v for v in available_versions if v != current_version]
            removable_candidates = [v for v in available_versions if v != current_version]

            st.markdown("#### Rollback to Previous Version")
            if pending_candidate:
                st.info(
                    "Rollback is disabled while a candidate is pending review. Approve or reject the candidate first."
                )
            elif rollback_candidates:
                rb_target = st.selectbox(
                    "Rollback target version",
                    rollback_candidates,
                    key="rollback_target_select",
                )
                confirm_rb = st.checkbox(
                    f"I confirm rollback deployment to {rb_target}.",
                    key="confirm_rollback_checkbox",
                )
                rollback_clicked = st.button(
                    ":material/restore: Rollback to Selected Version",
                    width="stretch",
                    disabled=not confirm_rb,
                    key="rollback_button",
                )
                if rollback_clicked:
                    try:
                        target, previous = _rollback_to_version(mt, rb_target, metrics)
                        st.cache_data.clear()
                        st.session_state[
                            "mlops_retrain_notice"
                        ] = f"Rollback completed: production moved from {previous} to {target}."
                        _queue_active_section(SECTION_VERSIONING)
                        st.rerun()
                    except Exception as exc:
                        st.error("Rollback failed. Review the exception below.")
                        st.exception(exc)
            else:
                st.info(
                    "No rollback candidates are available. Promote at least one additional version snapshot first."
                )

            st.markdown("#### Remove Version Snapshot")
            if pending_candidate:
                st.info(
                    "Version deletion is disabled while a candidate is pending review."
                )
            elif removable_candidates:
                delete_target = st.selectbox(
                    "Version to remove",
                    removable_candidates,
                    key="delete_version_select",
                )
                confirm_delete = st.checkbox(
                    f"I confirm permanent deletion of snapshot {delete_target}.",
                    key="confirm_delete_version_checkbox",
                )
                delete_clicked = st.button(
                    ":material/delete: Delete Selected Version Snapshot",
                    width="stretch",
                    disabled=not confirm_delete,
                    key="delete_version_button",
                )
                if delete_clicked:
                    try:
                        _remove_version_snapshot(delete_target, current_version)
                        st.cache_data.clear()
                        st.session_state[
                            "mlops_retrain_notice"
                        ] = f"Version snapshot {delete_target} was removed successfully."
                        _queue_active_section(SECTION_VERSIONING)
                        st.rerun()
                    except Exception as exc:
                        st.error("Version deletion failed. Review the exception below.")
                        st.exception(exc)
            else:
                st.info("No removable snapshots available. Active version cannot be deleted.")
        else:
            st.info(
                "No registry history found yet. Trigger a retrain from this page to create versioned records."
            )
            init_registry = st.button(
                ":material/history: Initialize Registry from Current Model",
                width="stretch",
                key="init_registry_button",
            )
            if init_registry:
                try:
                    version_paths = _snapshot_production_version(mt, "v1.0.0")
                except Exception as exc:
                    st.error("Registry initialization failed while snapshotting production artefacts.")
                    st.exception(exc)
                    version_paths = None

                if version_paths is None:
                    return

                init_record = {
                    "version": "v1.0.0",
                    "timestamp_utc": _utc_now_iso(),
                    "trigger": "bootstrap_existing_artefacts",
                    "snapshot_dir": version_paths["base_dir"],
                    "runtime_sec": 0.0,
                    "rows": int(len(df_full)),
                    "features": int(len(feat_cols)) if feat_cols else None,
                    "accuracy": round(metrics["accuracy"], 6),
                    "macro_f1": round(metrics["macro_f1"], 6),
                    "macro_recall": round(metrics["macro_recall"], 6),
                    "dataset_path": DATA_PATH,
                    "dataset_sha256": _sha256_of_file(DATA_PATH)[:16],
                    "dataset_fingerprint": _dataset_fingerprint(df_full)[:16],
                    "python_version": platform.python_version(),
                }
                _save_registry([init_record])
                st.cache_data.clear()
                st.session_state[
                    "mlops_retrain_notice"
                ] = "Registry initialized from current model artefacts as version v1.0.0."
                _queue_active_section(SECTION_VERSIONING)
                st.rerun()

    if active_section == SECTION_RETRAIN:
        st.markdown("#### Data Refresh")
        st.caption(
            "Simulates re-ingestion from multiple upstream systems and a refreshed aggregated dataset."
        )

        last_etl_report = st.session_state.get(ETL_REPORT_KEY)
        if last_etl_report:
            st.info(
                "Last ETL refresh: "
                f"{last_etl_report.get('refreshed_utc', 'Unknown')}"
            )
            etl_report_df = pd.DataFrame([last_etl_report])
            show_cols = [
                "refreshed_utc",
                "rows",
                "columns",
                "duplicates",
                "missing_cells",
                "dataset_sha256",
                "dataset_fingerprint",
            ]
            etl_show_cols = [c for c in show_cols if c in etl_report_df.columns]
            st.dataframe(etl_report_df[etl_show_cols], width="stretch")

        rerun_etl_clicked = st.button(
            ":material/sync: Re-run ETL",
            width="stretch",
            key="rerun_etl_button",
        )
        if rerun_etl_clicked:
            try:
                with st.status(
                    ":material/hub: Running ETL orchestration simulation ...",
                    expanded=True,
                ) as etl_status:
                    st.write("Step 1/6 - Connect to Student Information System feed")
                    st.write("Step 2/6 - Pull academic outcomes and performance feed")
                    st.write("Step 3/6 - Pull finance and attendance feed")
                    st.write("Step 4/6 - Join sources and standardize schema")
                    st.write("Step 5/6 - Run data quality checks")
                    etl_report, etl_logs = _simulate_etl_refresh(DATA_PATH)
                    st.write(
                        "Step 6/6 - Publish refreshed curated dataset "
                        f"({etl_report['rows']:,} rows, {etl_report['columns']} columns)"
                    )
                    etl_status.update(
                        label=":material/check_circle: ETL refresh completed.",
                        state="complete",
                    )

                st.session_state[ETL_REPORT_KEY] = etl_report
                st.session_state[ETL_LOGS_KEY] = etl_logs
                st.session_state[
                    "mlops_retrain_notice"
                ] = "ETL refresh completed. Dataset snapshot is ready for candidate retraining."
                _queue_active_section(SECTION_RETRAIN)
                st.rerun()
            except Exception as exc:
                st.error("ETL refresh failed. Review the exception below.")
                st.exception(exc)

        etl_logs = st.session_state.get(ETL_LOGS_KEY)
        if etl_logs:
            with st.expander("Latest ETL Run Log", expanded=False):
                _render_color_logs(etl_logs, max_lines=300)

        st.markdown("#### Candidate Retrain Workflow")
        st.caption(
            "Step 1: Configure feature/model/hyperparameter choices. "
            "Step 2: Train candidate. Step 3: Review and approve promotion."
        )

        st.info(
            "Preprocessing is automatic at retrain time: numeric columns are imputed, and "
            "categorical columns are encoded according to column type and observed distribution."
        )

        source_frame = source_df if source_df is not None else df_full
        all_feature_cols = [c for c in source_frame.columns if c != "Target"]
        blacklisted_features = [c for c in all_feature_cols if _is_blacklisted_feature(c, mt)]
        allowed_features = [c for c in all_feature_cols if c not in blacklisted_features]

        default_features = [c for c in (feat_cols or []) if c in allowed_features]
        if not default_features:
            default_features = allowed_features

        st.markdown("##### Feature Selection")
        if not allowed_features:
            st.error("No trainable features available after blacklist filtering.")
            return

        selected_features = st.multiselect(
            "Features to include in retraining",
            options=allowed_features,
            default=default_features,
            key="retrain_selected_features",
            help="Blacklisted leakage features are blocked automatically and cannot be selected.",
        )

        st.text_area(
            "Blocked/Blacklisted features (auto-excluded)",
            value="\n".join(blacklisted_features) if blacklisted_features else "None detected in current data.csv",
            height=140,
            disabled=True,
            key="retrain_blocked_features",
        )

        st.markdown("##### Model Selection")
        model_options = ["xgb", "lgbm", "catb"]
        selected_models = st.multiselect(
            "Base models to retrain",
            options=model_options,
            default=model_options,
            key="retrain_selected_models",
            format_func=lambda x: {"xgb": "XGBoost", "lgbm": "LightGBM", "catb": "CatBoost"}.get(x, x),
        )

        if len(selected_models) >= 2:
            st.success("Ensembler mode: a stacking meta-learner will be created.")
        elif len(selected_models) == 1:
            st.info("Single-model mode: no meta-learner will be created.")
        else:
            st.warning("Select at least one base model.")

        st.markdown("##### Hyperparameters (JSON)")
        default_params = {
            "xgb": getattr(mt, "XGB_PARAMS", {}),
            "lgbm": getattr(mt, "LGBM_PARAMS", {}),
            "catb": getattr(mt, "CATB_PARAMS", {}),
        }

        parsed_model_params = {}
        hp_errors = []
        for model_name in selected_models:
            hp_text = st.text_area(
                f"{model_name.upper()} hyperparameters",
                value=json.dumps(default_params.get(model_name, {}), indent=2),
                key=f"retrain_hp_json_{model_name}",
                height=220,
            )
            try:
                parsed = json.loads(hp_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Hyperparameters must be a JSON object.")
                parsed_model_params[model_name] = parsed
            except Exception as exc:
                hp_errors.append(f"{model_name.upper()}: {exc}")

        for err in hp_errors:
            st.error(err)

        if pending_candidate:
            st.info(
                "A pending candidate already exists. Approve or reject it before starting a new retrain run."
            )

        confirm_candidate_retrain = st.checkbox(
            "I confirm that I want to run retraining for review only (no immediate promotion).",
            key="confirm_candidate_retrain",
        )

        can_run_retrain = (
            confirm_candidate_retrain
            and pending_candidate is None
            and len(selected_features) > 0
            and len(selected_models) > 0
            and len(hp_errors) == 0
        )

        candidate_retrain_clicked = st.button(
            ":material/play_circle: Run Candidate Retrain",
            type="primary",
            width="stretch",
            disabled=not can_run_retrain,
            key="run_candidate_retrain_button",
        )

        if candidate_retrain_clicked:
            if not os.path.exists(DATA_PATH):
                st.error("Candidate retrain aborted: data.csv is missing.")
            else:
                try:
                    with st.status(
                        ":material/model_training: Running candidate retraining pipeline ...",
                        expanded=True,
                    ) as retrain_status:
                        st.write("Step 1/6 - Validate dataset availability and schema")
                        st.write("Step 2/6 - Apply leakage blacklist and feature configuration")
                        st.write("Step 3/6 - Build model pipelines and hyperparameter settings")
                        st.write("Step 4/6 - Train base learners and generate stacked signals")
                        st.write("Step 5/6 - Fit final candidate artefacts and score dataset")
                        candidate_info = _train_candidate_model(
                            mt,
                            DATA_PATH,
                            selected_features=selected_features,
                            selected_models=selected_models,
                            model_params=parsed_model_params,
                        )
                        st.write(
                            "Step 6/6 - Stage candidate for review "
                            f"(runtime: {candidate_info.get('runtime_sec', 0)} sec)"
                        )

                        trainer_logs = (candidate_info.get("trainer_logs") or "").strip()
                        if trainer_logs:
                            st.write("Trainer output (last 20 lines):")
                            tail_lines = trainer_logs.splitlines()[-20:]
                            _render_color_logs(tail_lines, max_lines=20)

                        retrain_status.update(
                            label=":material/check_circle: Candidate training completed and staged.",
                            state="complete",
                        )

                        candidate_info["baseline_metrics"] = {
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "macro_recall": metrics["macro_recall"],
                        }
                        candidate_info["blocked_features"] = blacklisted_features
                        st.session_state[PENDING_CANDIDATE_KEY] = candidate_info
                        st.session_state[RETRAIN_LOGS_KEY] = candidate_info.get("trainer_logs", "")
                        st.session_state[
                            "mlops_retrain_notice"
                        ] = (
                            f"Candidate {candidate_info['candidate_id']} trained successfully. "
                            "Review its results, then approve promotion if acceptable."
                        )
                    _queue_active_section(SECTION_RETRAIN)
                    st.rerun()
                except Exception as exc:
                    st.error("Candidate retrain failed. Review the exception below.")
                    st.exception(exc)

        retrain_logs = (st.session_state.get(RETRAIN_LOGS_KEY) or "").strip()
        if retrain_logs:
            with st.expander("Latest Candidate Retrain Log", expanded=False):
                _render_color_logs(retrain_logs, max_lines=600)

        pending_candidate = st.session_state.get(PENDING_CANDIDATE_KEY)
        if pending_candidate:
            st.markdown("#### Pending Candidate Review")
            st.warning(
                f"Candidate {pending_candidate.get('candidate_id', 'unknown')} is not promoted yet. "
                "Review the results below before approving promotion."
            )

            selected_models_text = ", ".join([m.upper() for m in pending_candidate.get("selected_models", [])])
            config_left, config_right = st.columns(2)
            config_left.metric("Configured Features", len(pending_candidate.get("selected_features", [])))
            config_right.metric("Configured Models", len(pending_candidate.get("selected_models", [])))
            st.caption(f"Configured base models: {selected_models_text if selected_models_text else 'Unknown'}")

            candidate_paths = pending_candidate.get("paths", {})
            candidate_scores_path = candidate_paths.get("scores_path", "")
            candidate_df = _safe_read_csv(candidate_scores_path) if candidate_scores_path else None

            if candidate_df is None or not {"Target", "Predicted_Target"}.issubset(candidate_df.columns):
                st.error(
                    "Candidate metrics could not be loaded from staged artefacts. "
                    "Reject this candidate and run retraining again."
                )
            else:
                candidate_metrics = _metrics_from_predictions(
                    candidate_df["Target"], candidate_df["Predicted_Target"]
                )
                pending_candidate["metrics"] = {
                    k: _safe_float(v) for k, v in candidate_metrics.items()
                }
                pending_candidate["rows"] = int(len(candidate_df))
                st.session_state[PENDING_CANDIDATE_KEY] = pending_candidate

                review_df = pd.DataFrame(
                    [
                        {
                            "Metric": "Accuracy",
                            "Production": metrics["accuracy"],
                            "Candidate": candidate_metrics["accuracy"],
                            "Delta": candidate_metrics["accuracy"] - metrics["accuracy"],
                        },
                        {
                            "Metric": "Macro F1",
                            "Production": metrics["macro_f1"],
                            "Candidate": candidate_metrics["macro_f1"],
                            "Delta": candidate_metrics["macro_f1"] - metrics["macro_f1"],
                        },
                        {
                            "Metric": "Macro Recall",
                            "Production": metrics["macro_recall"],
                            "Candidate": candidate_metrics["macro_recall"],
                            "Delta": candidate_metrics["macro_recall"] - metrics["macro_recall"],
                        },
                    ]
                )
                st.dataframe(
                    review_df.style.format(
                        {"Production": "{:.4f}", "Candidate": "{:.4f}", "Delta": "{:+.4f}"}
                    ),
                    width="stretch",
                )

                cand_report = classification_report(
                    candidate_df["Target"],
                    candidate_df["Predicted_Target"],
                    output_dict=True,
                    zero_division=0,
                )
                cand_report_df = pd.DataFrame(cand_report).transpose().rename(index=OUTCOME_LABEL_MAP)

                report_col1, report_col2 = st.columns(2)
                with report_col1:
                    st.markdown("##### Production Classification Report")
                    st.dataframe(report_df, width="stretch")
                with report_col2:
                    st.markdown("##### Candidate Classification Report")
                    st.dataframe(cand_report_df, width="stretch")

                staged_rows = [
                    _file_descriptor(candidate_paths.get("model_path", "")),
                    _file_descriptor(candidate_paths.get("label_encoder_path", "")),
                    _file_descriptor(candidate_paths.get("feature_cols_path", "")),
                    _file_descriptor(candidate_paths.get("thresholds_path", "")),
                    _file_descriptor(candidate_paths.get("scores_path", "")),
                ]
                st.markdown("##### Candidate Artefact Inventory")
                st.dataframe(pd.DataFrame(staged_rows), width="stretch")

                approve_promotion = st.checkbox(
                    "I reviewed the candidate results and approve promotion to production.",
                    key="approve_candidate_checkbox",
                )
                promote_col, reject_col = st.columns(2)
                approve_clicked = promote_col.button(
                    ":material/publish: Approve and Promote Candidate",
                    type="primary",
                    width="stretch",
                    disabled=not approve_promotion,
                    key="approve_promote_button",
                )
                reject_clicked = reject_col.button(
                    ":material/cancel: Reject Candidate",
                    width="stretch",
                    key="reject_candidate_button",
                )

                if approve_clicked:
                    try:
                        version = _promote_candidate_model(mt, pending_candidate, metrics)
                        st.session_state.pop(PENDING_CANDIDATE_KEY, None)
                        st.cache_data.clear()
                        st.session_state[
                            "mlops_retrain_notice"
                        ] = (
                            f"Candidate {pending_candidate.get('candidate_id', 'unknown')} approved and "
                            f"promoted to production as version {version}."
                        )
                        _queue_active_section(SECTION_RETRAIN)
                        st.rerun()
                    except Exception as exc:
                        st.error("Candidate promotion failed. Review the exception below.")
                        st.exception(exc)

                if reject_clicked:
                    _discard_candidate_model(pending_candidate)
                    st.session_state.pop(PENDING_CANDIDATE_KEY, None)
                    st.session_state[
                        "mlops_retrain_notice"
                    ] = "Candidate rejected. Production model remains unchanged."
                    _queue_active_section(SECTION_RETRAIN)
                    st.rerun()
