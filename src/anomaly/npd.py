import numpy as np
import pandas as pd
import h3

from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from src.config import AnomalyConfig

_DEFAULTS = AnomalyConfig()
K_VALUE = _DEFAULTS.k_value
TOP_PERCENTILE = _DEFAULTS.top_percentile


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.clip(v, 0, None)
    return v / (v.sum() + eps)


def build_full_profile(hourly_matrix, weekly_matrix, static_matrix) -> np.ndarray:
    """Full row-wise L2-normalized profile (hourly + weekly + static)."""
    H = np.array([_safe_normalize(r) for r in hourly_matrix])
    W = np.array([_safe_normalize(r[:7]) for r in weekly_matrix])
    S = MinMaxScaler().fit_transform(static_matrix)
    X = np.concatenate([H, W, S], axis=1).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-9)


def compute_npd(
    final_emb:    np.ndarray,
    full_profile: np.ndarray,
    hex_ids:      list,
    hex_to_alcaldia: dict,
    hex_crime_count: dict,
    k:            int,
    top_percentile: float = TOP_PERCENTILE,
) -> pd.DataFrame:
    """
    Compute the Neighbor Profile Divergence (NPD) for each hexagon.

    For each hexagon i:
        1. Find its k nearest neighbors in final_emb (cosine).
        2. Estimate the expected profile as a similarity-weighted average.
        3. Compute cosine distance between the real and expected profile.

    NPD in [0, 1]: 0 = no discordance, 1 = maximum discordance.
    """
    N = len(final_emb)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(final_emb)
    distances, indices = nn.kneighbors(final_emb)

    npd_scores     = np.zeros(N, dtype=np.float32)
    expected_count = np.zeros(N, dtype=np.float32)

    counts = np.array([hex_crime_count.get(h, 0) for h in hex_ids],
                      dtype=np.float32)

    for i in range(N):
        nbr_idx = indices[i, 1:]
        nbr_sim = np.clip(1.0 - distances[i, 1:], 0, None)
        nbr_sim /= (nbr_sim.sum() + 1e-9)

        expected_profile   = np.average(full_profile[nbr_idx],
                                        weights=nbr_sim, axis=0)
        npd_scores[i]      = float(1.0 - np.dot(full_profile[i],
                                                  expected_profile))
        expected_count[i]  = float(np.dot(nbr_sim, counts[nbr_idx]))

    threshold    = np.percentile(npd_scores, (1 - top_percentile) * 100)
    is_candidate = npd_scores >= threshold

    return pd.DataFrame({
        "hex_id":          hex_ids,
        "npd":             npd_scores,
        "real_count":      counts.astype(int),
        "expected_count":  expected_count.astype(int),
        "count_ratio":     counts / (expected_count + 1),
        "is_candidate":    is_candidate,
        "alcaldia":        [hex_to_alcaldia.get(h, "N/A") for h in hex_ids],
    }).sort_values("npd", ascending=False).reset_index(drop=True)


def sensitivity_analysis(
    final_emb:    np.ndarray,
    full_profile: np.ndarray,
    hex_ids:      list,
    hex_to_alcaldia: dict,
    hex_crime_count: dict,
    k_values:     list,
    k_ref:        int = K_VALUE,
    top_percentile: float = TOP_PERCENTILE,
) -> pd.DataFrame:
    """
    Compute NPD for each k in k_values and measure:
        - Ranking correlation between k and k_ref
        - Top 5% overlap between k and k_ref

    Returns a DataFrame with one row per value of k.
    """
    N         = len(final_emb)
    top_n     = max(1, int(N * top_percentile))

    # Compute NPD for k_ref first
    df_ref   = compute_npd(final_emb, full_profile, hex_ids,
                           hex_to_alcaldia, hex_crime_count, k_ref, top_percentile)
    top_ref  = set(df_ref.head(top_n)["hex_id"])
    rank_ref = df_ref.set_index("hex_id")["npd"]

    rows = []
    for k in k_values:
        df_k     = compute_npd(final_emb, full_profile, hex_ids,
                               hex_to_alcaldia, hex_crime_count, k, top_percentile)
        top_k    = set(df_k.head(top_n)["hex_id"])
        rank_k   = df_k.set_index("hex_id")["npd"]

        # Align by hex_id for correlation
        common   = rank_ref.index.intersection(rank_k.index)
        rho, _   = spearmanr(rank_ref[common].values, rank_k[common].values)
        overlap  = len(top_ref & top_k) / top_n

        rows.append({
            "k":             k,
            "k_ref":         k == k_ref,
            "spearman_rho":  rho,
            "top5_overlap":  overlap,
            "npd_mean":      df_k["npd"].mean(),
            "npd_std":       df_k["npd"].std(),
        })

    return pd.DataFrame(rows)


def table_top_candidates(df_npd: pd.DataFrame, n: int = 20, order="asc") -> pd.DataFrame:
    """
    Print a table with the top n anomaly candidates ranked by NPD.
    """
    top = df_npd.head(n).copy() if order == "asc" else df_npd.tail(n).copy()

    coords     = [h3.cell_to_latlng(h) for h in top["hex_id"]]
    top["lat"] = [round(c[0], 6) for c in coords]
    top["lon"] = [round(c[1], 6) for c in coords]
    top["count_ratio"] = top["count_ratio"].round(3)

    cols = ["hex_id", "alcaldia", "lat", "lon",
            "npd", "real_count", "expected_count", "count_ratio"]
    top  = top[cols]

    formatters = {
        "lat":            "{:.6f}".format,
        "lon":            "{:.6f}".format,
        "npd":            "{:.4f}".format,
        "real_count":     "{:,}".format,
        "expected_count": "{:,}".format,
        "count_ratio":    "{:.3f}".format,
    }
    with pd.option_context("display.max_colwidth", 28, "display.width", 120):
        print(top.to_string(index=False, formatters=formatters))
    return top
