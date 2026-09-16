"""Ejecuta el pipeline completo end-to-end y guarda figuras/métricas reales."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_processing import generate_synthetic_data, preprocess
from model import train_isolation_forest, score_anomalies, save_model
from evaluate import compute_metrics, plot_score_distribution, plot_flagged_scatter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figuras"
FIGURES.mkdir(parents=True, exist_ok=True)


def main():
    df = generate_synthetic_data(n_samples=5000, anomaly_rate=0.02, seed=42)
    df.to_csv(ROOT / "data" / "synthetic_transactions.csv", index=False)

    X, _ = preprocess(df)
    model = train_isolation_forest(X, contamination=0.02)
    scores, labels = score_anomalies(model, X)

    y_true = df["is_anomaly"].astype(int).to_numpy()
    metrics = compute_metrics(y_true, labels)

    plot_score_distribution(scores, y_true, save_path=str(FIGURES / "score_distribution.png"))
    plot_flagged_scatter(df, labels, y_true, save_path=str(FIGURES / "flagged_scatter.png"))

    save_model(model, str(RESULTS / "isolation_forest_model.joblib"))

    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
