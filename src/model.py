"""Entrenamiento y scoring del modelo de detección de anomalías (Isolation Forest)."""
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X, contamination: float = 0.02, random_state: int = 42) -> IsolationForest:
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    model.fit(X)
    return model


def score_anomalies(model: IsolationForest, X):
    """Devuelve (scores continuos, etiquetas binarias 0/1) donde 1 = anómalo."""
    scores = model.decision_function(X)
    raw_pred = model.predict(X)  # -1 = anómalo, 1 = normal
    labels = np.where(raw_pred == -1, 1, 0)
    return scores, labels


def save_model(model: IsolationForest, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> IsolationForest:
    return joblib.load(path)
