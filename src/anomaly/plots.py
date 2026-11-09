import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
import contextily as ctx
import xyzservices
from shapely.geometry import Polygon

from pathlib import Path
import h3

from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from src.config import AnomalyConfig, PROJECT_ROOT
from src.common.log import get_logger, print_table

logger = get_logger(__name__)

K_VALUE = AnomalyConfig().k_value


def _h3_to_polygon(h):
    boundary = h3.cell_to_boundary(h)
    return Polygon([(lon, lat) for lat, lon in boundary])


def _carto_api_key():
    key = os.environ.get("CARTO_API_KEY", "").strip()
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("CARTO_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def basemap(ax, zoom: int, alpha: float = 0.4) -> None:
    """CARTO Positron basemap; drawn only when a CARTO API key is available."""
    key = _carto_api_key()
    if key is None:
        logger.warning(
            "CARTO_API_KEY not set: map rendered WITHOUT basemap. Get a key at "
            "https://carto.com/ and add CARTO_API_KEY=<key> to .env at the repo root "
            "(or export it), then rerun."
        )
        return
    ctx.add_basemap(ax, source=_carto_positron(key), zoom=zoom, alpha=alpha)


def _carto_positron(api_key: str) -> xyzservices.TileProvider:
    """Keyed CARTO raster endpoint; the bundled CartoDB provider has no key placeholder."""
    return xyzservices.TileProvider(
        name="CartoDB.Positron",
        url="https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png?key={api_key}",
        attribution="(C) OpenStreetMap contributors (C) CARTO",
        api_key=api_key,
        max_zoom=20,
    )


def plot_sensitivity(df_npd:   pd.DataFrame,
                     df_sens:  pd.DataFrame,
                     k_ref:    int,
                     fig_dir:  Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ks      = df_sens["k"].values
    rho     = df_sens["spearman_rho"].values
    overlap = df_sens["top5_overlap"].values

    l1, = ax.plot(ks, rho, "o-", color="#377EB8", lw=2,
                  label="Spearman ρ (ranking)")
    l2, = ax2.plot(ks, overlap, "s--", color="#E41A1C", lw=2,
                   label="Top-5% overlap")

    k_ref_idx = list(ks).index(k_ref) if k_ref in ks else None
    if k_ref_idx is not None:
        ax.axvline(k_ref, color="gray", linestyle=":",
                   lw=1.5, label=f"k_ref=√N={k_ref}")
        ax.scatter([k_ref], [rho[k_ref_idx]],
                   s=100, color="#377EB8", zorder=5)
        ax2.scatter([k_ref], [overlap[k_ref_idx]],
                    s=100, color="#E41A1C", zorder=5)

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Spearman correlation", color="#377EB8")
    ax2.set_ylabel("Top-5% Overlap", color="#E41A1C")
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    lines = [l1, l2]
    if k_ref_idx is not None:
        lines += [plt.Line2D([0],[0], color="gray", linestyle=":",
                              label=f"k_ref=√N={k_ref}")]
    ax.legend(handles=lines, fontsize=8, loc="lower right")

    print("\nSensitivity analysis results:")
    headers = ["k", "Spearman rho", "Top-5% overlap"]
    rows = [
        [str(row.k), f"{row.spearman_rho:.4f}", f"{row.top5_overlap:.4f}"]
        for row in df_sens.itertuples()
    ]
    print_table(headers, rows)

    plt.tight_layout()
    plt.savefig(fig_dir / "anomaly_detection_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_npd_map(df_npd: pd.DataFrame, fig_dir: Path):
    """
    Map of Mexico City with hexagons colored by NPD over a real basemap.
    Candidates (top 5%) marked with a white outline.
    """
    df_map = df_npd.copy()
    df_map["geometry"] = df_map["hex_id"].apply(_h3_to_polygon)

    gdf = gpd.GeoDataFrame(df_map, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=(10, 11))

    norm = Normalize(vmin=gdf["npd"].min(), vmax=gdf["npd"].max())
    cmap = plt.cm.YlOrRd

    normal = gdf[~gdf["is_candidate"]]
    normal.plot(
        ax=ax, column="npd", cmap=cmap, norm=norm,
        alpha=0.6, edgecolor="none"
    )

    cands = gdf[gdf["is_candidate"]]
    cands.plot(
        ax=ax, column="npd", cmap=cmap, norm=norm,
        alpha=1.0, edgecolor="black", linewidth=0.8
    )

    basemap(ax, zoom=11)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("NPD Score", fontsize=10)

    ax.set_axis_off()
    plt.tight_layout()
    out = fig_dir / "npd_map.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_anomaly_profile(
    top_n:          int,
    df_npd:         pd.DataFrame,
    final_emb:      np.ndarray,
    hourly_matrix:  np.ndarray,
    weekly_matrix:  np.ndarray,
    hex_ids:        list,
    fig_dir:        Path,
    k_value:        int = K_VALUE,
):
    """
    Analysis figure for the top_n-th hexagon with highest NPD, showing:
      (A) Map: hex + its k_value neighbors in Mexico City
      (B) Histogram of real_count with the anomaly marked
      (C) Hourly distribution: hex vs neighbors
      (D) Weekly distribution: hex vs neighbors
    """
    hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    top_row  = df_npd.iloc[top_n - 1]
    top_id   = top_row["hex_id"]
    top_idx  = hex_to_idx[top_id]

    nn = NearestNeighbors(n_neighbors=k_value + 1, metric="cosine", n_jobs=-1)
    nn.fit(final_emb)
    _, indices = nn.kneighbors(final_emb[top_idx].reshape(1, -1))
    nbr_idxs = indices[0, 1:]                        # exclude the hex itself
    nbr_ids  = [hex_ids[i] for i in nbr_idxs]

    top_hourly  = hourly_matrix[top_idx]
    top_hourly  = top_hourly / (top_hourly.sum() + 1e-9)
    nbr_hourly  = hourly_matrix[nbr_idxs].mean(axis=0)
    nbr_hourly  = nbr_hourly / (nbr_hourly.sum() + 1e-9)

    top_weekly  = weekly_matrix[top_idx, :7]
    top_weekly  = top_weekly / (top_weekly.sum() + 1e-9)
    nbr_weekly  = weekly_matrix[nbr_idxs, :7].mean(axis=0)
    nbr_weekly  = nbr_weekly / (nbr_weekly.sum() + 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Map of the anomalous hexagon and its neighbors
    ax = axes[0, 0]
    all_ids    = [top_id] + nbr_ids
    roles      = ["anomaly"] + ["neighbor"] * k_value
    polys      = [_h3_to_polygon(h) for h in all_ids]
    gdf        = gpd.GeoDataFrame({"hex_id": all_ids, "role": roles},
                                   geometry=polys, crs="EPSG:4326")
    gdf        = gdf.to_crs("EPSG:3857")
    colors = {"neighbor": "#377EB8", "anomaly": "#E41A1C"}
    for role, color in colors.items():
        sub = gdf[gdf["role"] == role]
        sub.plot(ax=ax, color=color,
                 alpha=0.5 if role == "neighbor" else 0.9,
                 edgecolor="white", linewidth=0.6)
    basemap(ax, zoom=12)
    ax.set_axis_off()
    ax.set_title(f"(A) Top anomaly (red) and its {k_value} embedding neighbors (blue)")

    # (B) Histogram of real counts with the anomaly marked
    ax = axes[0, 1]
    nbr_counts = df_npd[df_npd["hex_id"].isin(nbr_ids)]["real_count"].values

    ax.boxplot(
        nbr_counts,
        labels=[f"Neighbors\n(n={k_value})"],
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#377EB8", alpha=0.6),
        medianprops=dict(color="black", lw=2),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#377EB8", alpha=0.5)
    )
    ax.axhline(top_row["real_count"], color="#E41A1C", lw=2,
               linestyle="--",
               label=f"Top anomaly = {int(top_row['real_count'])} incidents")
    ax.set_ylabel("Incident count")
    ax.set_title("(B) Incident count distribution")
    ax.legend(fontsize=9)

    # (C) Hourly distribution of the anomaly vs neighbors
    ax = axes[1, 0]
    hours = np.arange(24)
    ax.plot(hours, top_hourly, "o-", color="#E41A1C", lw=2,
            markersize=4, label="Top anomaly")
    ax.plot(hours, nbr_hourly, "o-", color="#377EB8", lw=2,
            markersize=4, label=f"Neighbors mean (n={k_value})")
    ax.fill_between(hours, top_hourly, nbr_hourly,
                    alpha=0.12, color="#999999")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Relative frequency")
    ax.set_title("(C) Hourly crime profile")
    ax.set_xticks(hours[::2])
    ax.legend(fontsize=9)

    # (D) Weekly distribution of the anomaly vs neighbors
    ax = axes[1, 1]
    x = np.arange(7)
    ax.plot(x, top_weekly, "o-", color="#E41A1C", lw=2,
            markersize=6, label="Top anomaly")
    ax.plot(x, nbr_weekly, "o-", color="#377EB8", lw=2,
            markersize=6, label=f"Neighbors mean (n={k_value})")
    ax.fill_between(x, top_weekly, nbr_weekly,
                    alpha=0.12, color="#999999")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Relative frequency")
    ax.set_title("(D) Weekly crime profile")
    ax.set_xticks(x)
    ax.set_xticklabels(DAYS)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = fig_dir / f"top_anomaly_profile_{top_n}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_distance_vs_npd(df_npd: pd.DataFrame, fig_dir: Path):
    """
    Scatter: distance of count_ratio to 1 (Y axis) vs NPD (X axis).
    """
    df = df_npd.copy()
    df["ratio_distance"] = (df["count_ratio"] - 1).abs()

    COLOR  = "#1A3A6B"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["npd"], df["ratio_distance"],
        color=COLOR, alpha=0.55, s=10, label="H3 cells"
    )

    # Global trend line
    from numpy.polynomial.polynomial import polyfit
    coef = polyfit(df["npd"], df["ratio_distance"], 1)
    x_line = np.linspace(df["npd"].min(), df["npd"].max(), 200)
    ax.plot(x_line, coef[0] + coef[1] * x_line,
            color="#333333", lw=2, linestyle="--",
            alpha=0.6, label="Linear Trend $y={:.3f}x + {:.3f}$".format(coef[1], coef[0]))

    # Spearman correlation
    rho, pval = spearmanr(df["npd"], df["ratio_distance"])
    logger.info(f"NPD vs count-ratio distance: Spearman rho={rho:.3f}, p={pval:.6f}")

    ax.set_xlabel("NPD score", fontsize=11)
    ax.set_ylabel("Distance to count ratio |1 - count_ratio|", fontsize=11)
    ax.set_title("NPD Score vs Distance to Count Ratio",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = fig_dir / "ratio_distance_vs_npd.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
