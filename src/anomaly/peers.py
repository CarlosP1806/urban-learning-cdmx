import numpy as np
import h3
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS_KM = 6371.0088


def peer_indices(final_emb: np.ndarray, k: int):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(final_emb)
    distances, indices = nn.kneighbors(final_emb)
    sims = np.clip(1.0 - distances[:, 1:], 0, None)
    return indices[:, 1:], sims


def similarity_stats(sims: np.ndarray) -> dict:
    flat = sims.ravel()
    return {
        "min":    float(flat.min()),
        "p05":    float(np.percentile(flat, 5)),
        "median": float(np.median(flat)),
        "mean":   float(flat.mean()),
        "max":    float(flat.max()),
    }


def cell_centroids(hex_ids: list) -> np.ndarray:
    return np.array([h3.cell_to_latlng(h) for h in hex_ids], dtype=float)


def pairwise_km(centroids: np.ndarray) -> np.ndarray:
    lat = np.radians(centroids[:, 0])[:, None]
    lon = np.radians(centroids[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def geo_nearest_indices(D: np.ndarray, k: int) -> np.ndarray:
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def random_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, k), dtype=int)
    for i in range(n):
        pool = np.delete(np.arange(n), i)
        out[i] = rng.choice(pool, k, replace=False)
    return out


def _grid_distance(a: str, b: str) -> int:
    try:
        return h3.grid_distance(a, b)
    except Exception:
        return -1


def describe_set(hex_ids: list, idx: np.ndarray, D: np.ndarray) -> dict:
    gd = np.array([[_grid_distance(hex_ids[i], hex_ids[j]) for j in row]
                   for i, row in enumerate(idx)])
    km = np.take_along_axis(D, idx, axis=1)
    return {
        "share_ring1":          float((gd == 1).mean()),
        "share_within_2_rings": float(((gd >= 1) & (gd <= 2)).mean()),
        "median_km":            float(np.median(km)),
        "mean_km":              float(km.mean()),
    }


def overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(idx_a, idx_b)]))


def median_peer_distance_km(idx: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.median(np.take_along_axis(D, idx, axis=1), axis=1)
