"""Dashboard Streamlit: detección de transacciones anómalas con Isolation Forest."""
import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_processing import generate_synthetic_data, preprocess
from model import train_isolation_forest, score_anomalies
from evaluate import compute_metrics

st.set_page_config(page_title="Detección de Anomalías", layout="wide")

st.title("Dashboard de Detección de Transacciones Anómalas")
st.markdown(
    "Detecta transacciones anómalas en un flujo sintético de pagos usando "
    "**Isolation Forest**. Ajusta el nivel de contaminación esperado en la barra lateral."
)

st.sidebar.header("Parámetros del modelo")
contamination = st.sidebar.slider("Contamination (tasa esperada de anomalías)", 0.005, 0.10, 0.02, 0.005)
n_samples = st.sidebar.slider("Número de transacciones sintéticas", 1000, 10000, 5000, 500)


@st.cache_data
def _get_data(n_samples: int, seed: int = 42):
    return generate_synthetic_data(n_samples=n_samples, anomaly_rate=0.02, seed=seed)


df = _get_data(n_samples)
X, _ = preprocess(df)
model = train_isolation_forest(X, contamination=contamination)
scores, labels = score_anomalies(model, X)

df = df.copy()
df["anomaly_score"] = scores
df["predicted_anomaly"] = labels

metrics = compute_metrics(df["is_anomaly"].astype(int), df["predicted_anomaly"])

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{metrics['precision']:.3f}")
col2.metric("Recall", f"{metrics['recall']:.3f}")
col3.metric("F1-score", f"{metrics['f1']:.3f}")

st.subheader("Transacciones marcadas como anómalas")
flagged = df[df["predicted_anomaly"] == 1].sort_values("anomaly_score")
st.dataframe(flagged, use_container_width=True)

st.subheader("Distribución de scores")
fig_hist = px.histogram(
    df, x="anomaly_score", color=df["is_anomaly"].map({True: "Anomalía real", False: "Normal"}),
    nbins=60, barmode="overlay", opacity=0.6,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Monto vs. hora (coloreado por predicción)")
fig_scatter = px.scatter(
    df, x="hour", y="amount",
    color=df["predicted_anomaly"].map({1: "Anómalo (predicho)", 0: "Normal (predicho)"}),
    symbol=df["is_anomaly"].map({True: "Anomalía real", False: "Normal real"}),
    color_discrete_map={"Anómalo (predicho)": "crimson", "Normal (predicho)": "seagreen"},
)
st.plotly_chart(fig_scatter, use_container_width=True)
