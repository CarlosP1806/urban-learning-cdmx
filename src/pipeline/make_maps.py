"""Maps of the processing steps and results, saved to src/outputs/figures/maps/.

Usage:
    python -m src.pipeline.make_maps
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

from src.config import ExperimentConfig
from src.common.io import load_raw_data, load_hex_features, load_embedding_artifacts
from src.common.log import get_logger, section, fmt_int
from src.features.hex_features import assign_hex_ids
from src.evaluation.metrics import cluster_labels, suggest_k_values
from src.anomaly.npd import build_full_profile, compute_npd
from src.anomaly.plots import _h3_to_polygon, basemap
from src.anomaly.peers import peer_indices, cell_centroids, pairwise_km, median_peer_distance_km

logger = get_logger(__name__)

DPI        = 180
FIGSIZE    = (10, 11)
ZOOM       = 11
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]


def _gdf(hex_ids, **columns) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {"hex_id": list(hex_ids), **columns},
        geometry=[_h3_to_polygon(h) for h in hex_ids],
        crs="EPSG:4326",
    )
    return gdf.to_crs("EPSG:3857")


def _finish(fig, ax, title, out):
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


def _choropleth(gdf, column, title, out, cmap="viridis", norm=None, label=None,
                alpha=0.75, cbar_ticks=None, cbar_ticklabels=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if norm is None:
        norm = Normalize(vmin=float(gdf[column].min()), vmax=float(gdf[column].max()))
    gdf.plot(ax=ax, column=column, cmap=cmap, norm=norm, alpha=alpha, edgecolor="none")
    basemap(ax, zoom=ZOOM)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(label or column, fontsize=10)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.set_ticklabels(cbar_ticklabels)
    _finish(fig, ax, title, out)


def map_incidents(df, cfg, out_dir):
    counts = df["hex_id"].value_counts()
    gdf = _gdf(counts.index, count=counts.values)
    gdf["kept"] = gdf["count"] >= cfg.features.min_events
    n_kept, n_removed = int(gdf["kept"].sum()), int((~gdf["kept"]).sum())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    norm = LogNorm(vmin=cfg.features.min_events, vmax=float(gdf["count"].max()))
    gdf[~gdf["kept"]].plot(ax=ax, color="#BBBBBB", alpha=0.9, edgecolor="#777777",
                           linewidth=0.3, hatch="////")
    gdf[gdf["kept"]].plot(ax=ax, column="count", cmap="YlOrRd", norm=norm,
                          alpha=0.75, edgecolor="none")
    basemap(ax, zoom=ZOOM)
    sm = ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Incidents per cell (log scale)", fontsize=10)
    ax.legend(handles=[Patch(facecolor="#BBBBBB", edgecolor="#777777", hatch="////",
                             label=f"Removed: < {cfg.features.min_events} incidents (n={n_removed})")],
              loc="upper left", fontsize=9, framealpha=0.9)
    _finish(fig, ax,
            f"Incidents per H3 cell (resolution {cfg.features.resolution}): "
            f"{n_kept} cells kept, {n_removed} removed",
            out_dir / "01_incidents_per_cell.png")


def map_views(hf, out_dir):
    C = hf.C
    gdf = _gdf(
        hf.hex_ids,
        night_share   = hf.hourly_matrix[:, NIGHT_HOURS].sum(axis=1),
        weekend_share = hf.weekly_matrix[:, 9],
        violent_share = hf.static_matrix[:, C],
        median_delay  = hf.static_matrix[:, C + 4],
    )
    _choropleth(gdf, "night_share",
                "Hourly view: share of incidents between 22:00 and 05:59",
                out_dir / "02_view_hourly_night_share.png", label="Night-time share")
    _choropleth(gdf, "weekend_share",
                "Weekly view: share of incidents on Saturday and Sunday",
                out_dir / "03_view_weekly_weekend_share.png", label="Weekend share")
    _choropleth(gdf, "violent_share",
                "Static view: share of incidents classified as violent",
                out_dir / "04_view_static_violent_share.png", label="Violent share")
    vmax = float(np.percentile(gdf["median_delay"], 99))
    _choropleth(gdf, "median_delay",
                "Static view: median delay between incident and report (days)",
                out_dir / "05_view_static_median_delay.png",
                norm=Normalize(vmin=0, vmax=vmax),
                label="Median reporting delay, days (colour clipped at 99th percentile)")


def map_clusters(hex_ids, final_emb, K, out_dir):
    labels = cluster_labels(final_emb, K)
    gdf = _gdf(hex_ids, cluster=labels)
    colors = ListedColormap((list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors))[:K])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    gdf.plot(ax=ax, column="cluster", cmap=colors, categorical=True, alpha=0.8,
             edgecolor="white", linewidth=0.2)
    basemap(ax, zoom=ZOOM)
    sizes = pd.Series(labels).value_counts().sort_index()
    handles = [Patch(facecolor=colors(i), label=f"{i} (n={sizes[i]})") for i in range(K)]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=3, framealpha=0.9,
              title=f"Cluster (K={K})", title_fontsize=8)
    _finish(fig, ax, f"K-means clusters of the final embedding, K={K}",
            out_dir / f"06_kmeans_clusters_K{K}.png")


def map_peer_examples(hex_ids, df_npd, peer_idx, D, out_dir):
    cent   = cell_centroids(hex_ids)
    dist_c = pairwise_km(np.vstack([cent.mean(axis=0), cent]))[0, 1:]
    hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
    alcaldia   = df_npd.set_index("hex_id")["alcaldia"]
    examples = [
        ("Highest-NPD cell", hex_to_idx[df_npd.iloc[0]["hex_id"]]),
        ("Central cell (closest to the centroid of all cells)", int(np.argmin(dist_c))),
        ("Peripheral cell (farthest from the centroid of all cells)", int(np.argmax(dist_c))),
    ]
    base = _gdf(hex_ids)
    k = peer_idx.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (label, i) in zip(axes, examples):
        role = np.full(len(hex_ids), "other", dtype=object)
        role[peer_idx[i]] = "peer"
        role[i] = "cell"
        g = base.assign(role=role)
        g[g.role == "other"].plot(ax=ax, color="#DDDDDD", alpha=0.5, edgecolor="none")
        g[g.role == "peer"].plot(ax=ax, color="#377EB8", alpha=0.85, edgecolor="white", linewidth=0.4)
        g[g.role == "cell"].plot(ax=ax, color="#E41A1C", alpha=0.95, edgecolor="black", linewidth=0.8)
        basemap(ax, zoom=ZOOM)
        ax.set_axis_off()
        h = hex_ids[i]
        ax.set_title(f"{label}\n{h} ({alcaldia.get(h, 'N/A')}), median peer distance "
                     f"{np.median(D[i, peer_idx[i]]):.1f} km", fontsize=10)
    fig.legend(handles=[Patch(facecolor="#E41A1C", label="Example cell"),
                        Patch(facecolor="#377EB8", label=f"Its {k} embedding peers"),
                        Patch(facecolor="#DDDDDD", label="Other cells")],
               loc="lower center", ncol=3, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = out_dir / "07_peers_examples.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


def map_peer_distance(hex_ids, peer_idx, D, out_dir):
    med = median_peer_distance_km(peer_idx, D)
    gdf = _gdf(hex_ids, median_km=med)
    _choropleth(gdf, "median_km",
                f"Median distance from each cell to its {peer_idx.shape[1]} embedding peers",
                out_dir / "08_median_peer_distance_km.png", label="Median peer distance (km)")


def map_npd(df_npd, out_dir):
    gdf = _gdf(df_npd["hex_id"], npd=df_npd["npd"].values, cand=df_npd["is_candidate"].values)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    norm = Normalize(vmin=float(gdf["npd"].min()), vmax=float(gdf["npd"].max()))
    gdf[~gdf["cand"]].plot(ax=ax, column="npd", cmap="YlOrRd", norm=norm, alpha=0.6, edgecolor="none")
    gdf[gdf["cand"]].plot(ax=ax, column="npd", cmap="YlOrRd", norm=norm, alpha=1.0,
                          edgecolor="black", linewidth=0.8)
    basemap(ax, zoom=ZOOM)
    sm = ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02).set_label("NPD", fontsize=10)
    ax.legend(handles=[Patch(facecolor="none", edgecolor="black",
                             label=f"Top-5% candidates (n={int(gdf['cand'].sum())})")],
              loc="upper left", fontsize=9, framealpha=0.9)
    _finish(fig, ax, "Neighbor Profile Divergence (k=32) with top-5% candidates outlined",
            out_dir / "09_npd_candidates.png")


def map_candidate_stability(final_emb, full_profile, hf, k_values, top_percentile, out_dir):
    counts = pd.Series(0, index=hf.hex_ids)
    for k in k_values:
        d = compute_npd(final_emb, full_profile, hf.hex_ids, hf.hex_to_alcaldia,
                        hf.hex_crime_count, k=k, top_percentile=top_percentile)
        counts = counts.add(d.set_index("hex_id")["is_candidate"].astype(int), fill_value=0)
    n_k = len(k_values)
    gdf = _gdf(hf.hex_ids, times=counts.loc[hf.hex_ids].values.astype(int))
    cmap = ListedColormap(["#F0F0F0"] + [plt.cm.YlOrRd(x) for x in np.linspace(0.25, 1.0, n_k)])
    norm = BoundaryNorm(np.arange(-0.5, n_k + 1), cmap.N)
    n_all = int((gdf["times"] == n_k).sum())
    _choropleth(gdf, "times",
                f"Candidate stability: number of k values (of {n_k}: {', '.join(map(str, k_values))}) "
                f"in which a cell is in the top 5% — {n_all} cells in all {n_k}",
                out_dir / "10_candidate_stability.png", cmap=cmap, norm=norm, alpha=0.8,
                label="Number of k values", cbar_ticks=list(range(n_k + 1)))


def map_count_ratio(df_npd, out_dir):
    ratio = df_npd["count_ratio"].values.astype(float)
    log_r = np.log10(np.clip(ratio, 1e-3, None))
    L = float(np.percentile(np.abs(log_r), 99))
    gdf = _gdf(df_npd["hex_id"], log_ratio=log_r)
    ticks = np.linspace(-L, L, 5)
    _choropleth(gdf, "log_ratio",
                "Observed / expected incident count per cell (expected = similarity-weighted peer mean)",
                out_dir / "11_count_ratio.png", cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-L, vmax=L),
                label="Observed / expected ratio (log scale, colour clipped at 99th percentile of |log ratio|)",
                cbar_ticks=ticks, cbar_ticklabels=[f"{10 ** t:.2g}" for t in ticks])


def main():
    t0  = time.time()
    cfg = ExperimentConfig()
    out_dir = cfg.paths.figures_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    section("Loading data, features and embeddings")
    df = load_raw_data(cfg)
    df["hex_id"] = assign_hex_ids(df, cfg.features.resolution)
    hf = load_hex_features(cfg, df=df)
    final_emb, hex_ids, meta = load_embedding_artifacts(cfg)
    assert hex_ids == hf.hex_ids, "saved embeddings and rebuilt features disagree on the cell set"
    logger.info(f"{fmt_int(hf.N)} hexagons, {hf.C} crime categories")

    k = cfg.anomaly.k_value
    full_profile = build_full_profile(hf.hourly_matrix, hf.weekly_matrix, hf.static_matrix)
    df_npd = compute_npd(final_emb, full_profile, hf.hex_ids, hf.hex_to_alcaldia,
                         hf.hex_crime_count, k=k, top_percentile=cfg.anomaly.top_percentile)
    peer_idx, _ = peer_indices(final_emb, k)
    D = pairwise_km(cell_centroids(hf.hex_ids))
    k_values = suggest_k_values(hf.N)
    K = k_values[len(k_values) // 2]

    section(f"Rendering maps to {out_dir}/")
    map_incidents(df, cfg, out_dir)
    map_views(hf, out_dir)
    map_clusters(hf.hex_ids, final_emb, K, out_dir)
    map_peer_examples(hf.hex_ids, df_npd, peer_idx, D, out_dir)
    map_peer_distance(hf.hex_ids, peer_idx, D, out_dir)
    map_npd(df_npd, out_dir)
    map_candidate_stability(final_emb, full_profile, hf, cfg.anomaly.k_values_sensitivity,
                            cfg.anomaly.top_percentile, out_dir)
    map_count_ratio(df_npd, out_dir)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
