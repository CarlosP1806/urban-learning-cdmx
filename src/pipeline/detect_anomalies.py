import warnings
warnings.filterwarnings("ignore")

from src.config import ExperimentConfig
from src.common.io import load_hex_features, load_embedding_artifacts
from src.common.log import get_logger, section, fmt_int
from src.anomaly.npd import build_full_profile, compute_npd, sensitivity_analysis, table_top_candidates
from src.anomaly.plots import (plot_sensitivity, plot_npd_map,
                               plot_ratio_distance_vs_npd, plot_top_anomaly_profile)

logger = get_logger(__name__)


def main():
    cfg = ExperimentConfig()
    fig_dir = cfg.paths.figures_dir / "anomaly_detection"
    fig_dir.mkdir(parents=True, exist_ok=True)

    final_emb, hex_ids, meta = load_embedding_artifacts(cfg)
    hf = load_hex_features(cfg)

    hourly_matrix   = hf.hourly_matrix
    weekly_matrix   = hf.weekly_matrix
    static_matrix   = hf.static_matrix
    hex_to_alcaldia = hf.hex_to_alcaldia
    hex_crime_count = hf.hex_crime_count

    k_ref = cfg.anomaly.k_value
    logger.info(f"{fmt_int(len(hex_ids))} hexagons loaded, k_ref={k_ref}, "
                f"top_percentile={cfg.anomaly.top_percentile:.0%}")

    section("Computing Neighbor Profile Divergence (NPD)")
    full_profile = build_full_profile(hourly_matrix, weekly_matrix, static_matrix)

    df_npd = compute_npd(
        final_emb, full_profile,
        hex_ids, hex_to_alcaldia, hex_crime_count,
        k=k_ref, top_percentile=cfg.anomaly.top_percentile,
    )
    n_candidates = int(df_npd["is_candidate"].sum())
    logger.info(f"{fmt_int(n_candidates)} candidate hexagons flagged "
                f"(NPD mean={df_npd['npd'].mean():.4f}, std={df_npd['npd'].std():.4f})")

    section("Sensitivity analysis across k")
    df_sens = sensitivity_analysis(
        final_emb, full_profile,
        hex_ids, hex_to_alcaldia, hex_crime_count,
        k_values=cfg.anomaly.k_values_sensitivity,
        k_ref=k_ref, top_percentile=cfg.anomaly.top_percentile,
    )

    section("Top candidates by NPD")
    table_top_candidates(df_npd, n=20)

    section("Most similar candidates (lowest NPD)")
    table_top_candidates(df_npd, n=20, order="desc")

    section("Rendering figures")
    plot_sensitivity(df_npd, df_sens, k_ref, fig_dir=fig_dir)
    plot_npd_map(df_npd, fig_dir=fig_dir)
    plot_ratio_distance_vs_npd(df_npd, fig_dir=fig_dir)

    for top_n in (1, 2, 3):
        plot_top_anomaly_profile(
            top_n, df_npd, final_emb,
            hourly_matrix, weekly_matrix,
            hex_ids, fig_dir=fig_dir, k_value=k_ref,
        )
    logger.info(f"Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
