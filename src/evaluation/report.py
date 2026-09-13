import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.evaluation.metrics import (cluster_labels, metric_silhouette,
                                    metric_calinski_harabasz, metric_davies_bouldin)
from src.common.log import print_table


def build_input_matrix(hourly_matrix, weekly_matrix, static_matrix):
    """
    Concatenate the three feature matrices into one.
    Applies a StandardScaler per block before concatenating,
    same as the main pipeline does before converting to tensors.

    Returns X_raw (scaled) and X_nonneg (min-max >= 0, for NMF).
    """
    sc = StandardScaler()
    H  = sc.fit_transform(hourly_matrix)
    W  = sc.fit_transform(weekly_matrix)
    S  = sc.fit_transform(static_matrix)
    X_raw = np.concatenate([H, W, S], axis=1).astype(np.float32)

    # NMF requires non-negative values
    mm = MinMaxScaler()
    X_nonneg = mm.fit_transform(X_raw).astype(np.float32)

    return X_raw, X_nonneg


def print_final_table(results: list[dict], k_values: list) -> None:
    """Print the final table with mean ± std for K-dependent metrics."""
    print(f"\nFinal embedding comparison (mean ± std over K={k_values})")

    headers = ["Method", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "Dim"]
    rows = [
        [
            r["name"],
            f"{r['sil_mean']:+.4f} ± {r['sil_std']:.4f}",
            f"{r['ch_mean']:,.1f} ± {r['ch_std']:,.1f}",
            f"{r['db_mean']:.4f} ± {r['db_std']:.4f}",
            str(r["dim"]),
        ]
        for r in results
    ]
    print_table(headers, rows, aligns=["l", "r", "r", "r", "r"])


def plot_per_k(embeddings: dict,
               names: dict,
               k_values: list,
               output_dir: Path = None) -> plt.Figure:
    """
    Build a figure with 3 panels (Silhouette, Calinski-Harabasz, Davies-Bouldin).
    Each group of bars shows all methods for that K.
    Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin.
    """
    methods   = list(embeddings.keys())
    n_methods = len(methods)
    n_k       = len(k_values)

    COLORS = ["#377EB8", "#4DAF4A", "#FF7F00", "#984EA3", "#E41A1C"]
    LABELS = [names[m].split("—")[-1].strip() for m in methods]

    data = {m: {"sil": [], "ch": [], "db": []} for m in methods}

    for k in k_values:
        for m, emb in embeddings.items():
            labels = cluster_labels(emb, k)
            data[m]["sil"].append(metric_silhouette(emb, labels))
            data[m]["ch"].append(metric_calinski_harabasz(emb, labels))
            data[m]["db"].append(metric_davies_bouldin(emb, labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bar_width = 0.8 / n_methods
    x = np.arange(n_k)

    def draw_panel(ax, metric_key, title):
        for i, (m, color, label) in enumerate(zip(methods, COLORS, LABELS)):
            offset = (i - n_methods / 2 + 0.5) * bar_width
            vals   = data[m][metric_key]
            ax.bar(x + offset, vals,
                  width=bar_width * 0.9,
                  color=color, alpha=0.85,
                  label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in k_values], fontsize=10)
        ax.set_title(f"{title}",
                     fontsize=11, fontweight="bold")
        ax.set_axisbelow(True)

    draw_panel(axes[0], "sil", "Silhouette Score")
    draw_panel(axes[1], "ch",  "Calinski-Harabasz")
    draw_panel(axes[2], "db",  "Davies-Bouldin Index")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_,
               loc="lower center", ncol=n_methods,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "metrics_per_k.png", bbox_inches="tight")

    return fig
