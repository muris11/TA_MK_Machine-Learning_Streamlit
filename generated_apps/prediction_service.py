from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR if (BASE_DIR / "ml_artifacts").exists() else BASE_DIR.parent
ARTIFACT_DIR = PROJECT_ROOT / "ml_artifacts"
MODEL_PATH = ARTIFACT_DIR / "poverty_model_bundle.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact tidak ditemukan: {MODEL_PATH}")

_bundle = joblib.load(MODEL_PATH)


def priority_from_prediction(value: float) -> str:
    thresholds = _bundle["priority_thresholds"]
    if value <= thresholds["low_threshold"]:
        return "Low Priority"
    if value <= thresholds["high_threshold"]:
        return "Medium Priority"
    return "High Priority"


def predict_condition(tahun: int, gini_ratio: float, tpt: float, inflasi: float, ipm: float) -> Dict[str, Any]:
    record = {
        "tahun": int(tahun),
        "gini_ratio": float(gini_ratio),
        "tingkat_penganggur_terbuka": float(tpt),
        "rata_rata_inflasi_tahunan": float(inflasi),
        "indeks_pembangunan_manusia": float(ipm),
    }
    frame = pd.DataFrame([record], columns=_bundle["feature_columns"])
    prediction = float(_bundle["regression_model"].predict(frame)[0])
    classifier_check = str(_bundle["classification_model"].predict(frame)[0])
    priority = priority_from_prediction(prediction)
    rule = _bundle["recommendation_rules"][priority]
    return {
        "input": record,
        "prediksi_kemiskinan": round(prediction, 2),
        "priority_level": priority,
        "prioritas_intervensi": priority,
        "classifier_priority_check": classifier_check,
        "status": rule["status"],
        "rekomendasi_utama": rule["main_recommendation"],
        "alasan": rule["reason"],
        "aksi_kebijakan": rule["policy_actions"],
        "timeline": rule["timeline"],
        "metadata": _bundle.get("model_metrics", {}),
    }
