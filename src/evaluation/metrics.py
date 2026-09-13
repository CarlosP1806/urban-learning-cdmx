import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score)

from src.config import EvalConfig

_DEFAULTS = EvalConfig()
RANDOM_STATE = _DEFAULTS.random_state


def cluster_labels(emb, n_clusters):
    """Run KMeans on the embedding to obtain cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                n_init="auto")
    return km.fit_predict(emb)


def metric_silhouette(emb, labels):
    """
    Silhouette Score in [-1, 1].
    Higher is better: 1 = perfectly separated clusters.
    Measures intra-cluster cohesion vs inter-cluster separation.
    """
    # Subsample if N is very large
    n = len(emb)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        return silhouette_score(emb[idx], labels[idx], metric="cosine")
    return silhouette_score(emb, labels, metric="cosine")


def metric_calinski_harabasz(emb, labels):
    """
    Calinski-Harabasz Index. Higher is better.
    Ratio between inter-cluster and intra-cluster dispersion.
    """
    return calinski_harabasz_score(emb, labels)


def metric_davies_bouldin(emb, labels):
    """
    Davies-Bouldin Index. Lower is better.
    Average similarity between each cluster and its closest neighbor.
    """
    return davies_bouldin_score(emb, labels)


def evaluate_embedding(name, emb, names, k_values):
    """
    Compute clustering metrics for each K in k_values and report
    mean ± standard deviation.
    """
    sils, chs, dbs = [], [], []
    for k in k_values:
        labels = cluster_labels(emb, k)
        sils.append(metric_silhouette(emb, labels))
        chs.append(metric_calinski_harabasz(emb, labels))
        dbs.append(metric_davies_bouldin(emb, labels))

    sil_mean, sil_std = np.mean(sils), np.std(sils)
    ch_mean,  ch_std  = np.mean(chs),  np.std(chs)
    db_mean,  db_std  = np.mean(dbs),  np.std(dbs)

    return {
        "name":            names[name],
        "sil_mean":        sil_mean,
        "sil_std":         sil_std,
        "ch_mean":         ch_mean,
        "ch_std":          ch_std,
        "db_mean":         db_mean,
        "db_std":          db_std,
        "k_values":        str(k_values),
        "dim":             emb.shape[1],
    }
