"""Generación y preprocesamiento de datos sintéticos de transacciones."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

N_LOCATIONS = 8


def generate_synthetic_data(n_samples: int = 5000, anomaly_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame sintético de transacciones con anomalías inyectadas.

    Columnas: transaction_id, amount, hour, location_code, frequency_last_24h, is_anomaly.
    """
    rng = np.random.default_rng(seed)
    n_anomalies = max(1, int(round(n_samples * anomaly_rate)))
    n_normal = n_samples - n_anomalies

    # --- Transacciones normales ---
    amount_normal = rng.lognormal(mean=3.2, sigma=0.6, size=n_normal)  # ~ USD 10-100
    # Horas concentradas en horario laboral (9-18) con algo de ruido fuera de ese rango.
    hour_business = rng.normal(loc=13.5, scale=3.2, size=n_normal)
    hour_normal = np.clip(hour_business, 0, 23.999)
    location_normal = rng.integers(0, N_LOCATIONS, size=n_normal)
    freq_normal = rng.poisson(lam=3, size=n_normal).astype(float)

    normal_df = pd.DataFrame({
        "amount": amount_normal,
        "hour": hour_normal,
        "location_code": location_normal,
        "frequency_last_24h": freq_normal,
        "is_anomaly": False,
    })

    # --- Anomalías: montos atípicamente altos o patrones de hora/frecuencia inusuales ---
    half = n_anomalies // 2
    rest = n_anomalies - half

    # Grupo 1: montos extremos, en horario normal.
    amount_high = rng.uniform(1500, 8000, size=half)
    hour_high = rng.uniform(0, 24, size=half)
    location_high = rng.integers(0, N_LOCATIONS, size=half)
    freq_high = rng.poisson(lam=3, size=half).astype(float)

    # Grupo 2: monto moderado pero patrón de hora/frecuencia inusual (madrugada, ráfagas).
    amount_odd = rng.lognormal(mean=3.5, sigma=0.7, size=rest)
    hour_odd = rng.uniform(1, 4.5, size=rest)  # madrugada
    location_odd = rng.integers(0, N_LOCATIONS, size=rest)
    freq_odd = rng.poisson(lam=25, size=rest).astype(float) + 15  # ráfaga de frecuencia

    anomaly_df = pd.DataFrame({
        "amount": np.concatenate([amount_high, amount_odd]),
        "hour": np.concatenate([hour_high, hour_odd]),
        "location_code": np.concatenate([location_high, location_odd]),
        "frequency_last_24h": np.concatenate([freq_high, freq_odd]),
        "is_anomaly": True,
    })

    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN-{i:06d}" for i in range(len(df))])
    df["hour"] = df["hour"].round(2)
    df["amount"] = df["amount"].round(2)
    df["frequency_last_24h"] = df["frequency_last_24h"].astype(int)
    return df


def preprocess(df: pd.DataFrame):
    """Escala amount/hour/frequency_last_24h con StandardScaler.

    Devuelve (X, scaler) donde X es un DataFrame numérico listo para el modelo.
    location_code se deja como columna categórica codificada (pocas categorías).
    """
    feature_cols = ["amount", "hour", "frequency_last_24h"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    X = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    X["location_code"] = df["location_code"].astype(int)
    return X, scaler
