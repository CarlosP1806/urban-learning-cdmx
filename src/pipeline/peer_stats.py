"""Peer similarity and peers-vs-geography statistics for the saved embedding.

Usage:
    python -m src.pipeline.peer_stats
"""

import warnings
warnings.filterwarnings("ignore")

from src.config import ExperimentConfig
from src.common.io import load_embedding_artifacts
from src.common.log import get_logger, section, print_table, fmt_int
from src.anomaly.peers import (peer_indices, similarity_stats, cell_centroids,
                               pairwise_km, geo_nearest_indices, random_indices,
                               describe_set, overlap)

logger = get_logger(__name__)


def main():
    cfg = ExperimentConfig()
    final_emb, hex_ids, meta = load_embedding_artifacts(cfg)
    k    = cfg.anomaly.k_value
    seed = cfg.eval.random_state
    logger.info(f"{fmt_int(len(hex_ids))} hexagons, k={k}, seed={seed}")

    section(f"Peer cosine similarity in embedding space (k={k})")
    peer_idx, sims = peer_indices(final_emb, k)
    st = similarity_stats(sims)
    print_table(["Statistic", "Value"],
                [[name, f"{st[name]:.6f}"] for name in ("min", "p05", "median", "mean", "max")],
                aligns=["l", "r"])

    section("Embedding peers versus geographic neighbours")
    D        = pairwise_km(cell_centroids(hex_ids))
    geo_idx  = geo_nearest_indices(D, k)
    rand_idx = random_indices(len(hex_ids), k, seed)
    sets = [
        (f"Embedding peers (k={k})",        peer_idx),
        (f"Geographically nearest (k={k})", geo_idx),
        (f"Random cells (k={k}, seed={seed})", rand_idx),
    ]
    rows = []
    for name, idx in sets:
        d = describe_set(hex_ids, idx, D)
        rows.append([name,
                     f"{100 * d['share_ring1']:.1f}%",
                     f"{100 * d['share_within_2_rings']:.1f}%",
                     f"{d['median_km']:.1f}",
                     f"{d['mean_km']:.1f}"])
    print_table(["Neighbour set", "Ring-1", "Within 2 rings", "Median km", "Mean km"],
                rows, aligns=["l", "r", "r", "r", "r"])
    ov = overlap(peer_idx, geo_idx)
    print(f"\nOverlap between embedding peers and the {k} geographically nearest cells: "
          f"{100 * ov:.1f}% (mean per cell)")


if __name__ == "__main__":
    main()
