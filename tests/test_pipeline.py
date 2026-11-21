"""Pruebas del pipeline de detección de anomalías (datos pequeños, semilla fija)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_processing import generate_synthetic_data, preprocess  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from model import score_anomalies, train_isolation_forest  # noqa: E402


def test_datos_sinteticos_tamano_columnas_y_tasa():
    df = generate_synthetic_data(n_samples=1000, anomaly_rate=0.02, seed=42)
    assert len(df) == 1000
    assert {"transaction_id", "amount", "hour", "location_code", "frequency_last_24h", "is_anomaly"} <= set(df.columns)
    assert 0.01 <= df["is_anomaly"].mean() <= 0.03


def test_datos_reproducibles_con_la_misma_semilla():
    a = generate_synthetic_data(n_samples=200, seed=7)
    b = generate_synthetic_data(n_samples=200, seed=7)
    assert a.equals(b)


def test_preprocess_escala_las_variables_continuas():
    df = generate_synthetic_data(n_samples=500, seed=1)
    X, _ = preprocess(df)
    for col in ["amount", "hour", "frequency_last_24h"]:
        assert abs(X[col].mean()) < 1e-6
        assert abs(X[col].std(ddof=0) - 1) < 1e-6


def test_modelo_detecta_la_mayoria_de_anomalias():
    df = generate_synthetic_data(n_samples=2000, anomaly_rate=0.02, seed=42)
    X, _ = preprocess(df)
    model = train_isolation_forest(X, contamination=0.02)
    scores, labels = score_anomalies(model, X)
    assert set(np.unique(labels)) <= {0, 1}
    assert len(scores) == len(df)
    m = compute_metrics(df["is_anomaly"].astype(int), labels)
    assert m["recall"] >= 0.8
    assert m["precision"] >= 0.8


def test_metricas_prediccion_perfecta():
    assert compute_metrics([0, 1, 1, 0], [0, 1, 1, 0]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
