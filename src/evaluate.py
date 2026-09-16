"""Métricas y visualizaciones para el detector de anomalías."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_score_distribution(scores, y_true, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(x=scores[~y_true.astype(bool)], color="steelblue", label="Normal", stat="density", kde=True, ax=ax, alpha=0.6)
    sns.histplot(x=scores[y_true.astype(bool)], color="crimson", label="Anomalía real", stat="density", kde=True, ax=ax, alpha=0.6)
    ax.set_xlabel("Score (decision_function)")
    ax.set_title("Distribución de scores de anomalía")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_flagged_scatter(df, y_pred, y_true, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["crimson" if p == 1 else "seagreen" for p in y_pred]
    ax.scatter(df["hour"], df["amount"], c=colors, s=18, alpha=0.6, label="_nolegend_")
    real_mask = y_true.astype(bool)
    ax.scatter(df.loc[real_mask, "hour"], df.loc[real_mask, "amount"], facecolors="none",
               edgecolors="black", s=80, linewidths=1.2, label="Anomalía real")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Monto")
    ax.set_title("Transacciones: predicción del modelo (rojo=anómalo, verde=normal)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
