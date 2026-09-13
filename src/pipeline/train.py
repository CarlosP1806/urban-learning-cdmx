import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from src.config import ExperimentConfig
from src.common.io import load_raw_data, load_hex_features, save_embedding_artifacts
from src.common.log import get_logger, section, fmt_int
from src.features.hex_features import to_tensors
from src.models.phase1 import build_pairs, train_phase1
from src.models.phase2 import train_phase2

logger = get_logger(__name__)


def main():
    cfg    = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    section("Loading data and building hex features")
    df = load_raw_data(cfg)
    hf = load_hex_features(cfg, df=df)
    logger.info(f"{fmt_int(hf.N)} hexagons, {fmt_int(len(df))} events, {hf.C} crime categories")
    T_h, T_w, T_s = to_tensors(hf, device)

    positives, negatives = build_pairs(
        hf.hex_ids, hf.hex_to_idx,
        k_ring=cfg.phase1.k_ring, n_neg=cfg.phase1.n_neg,
    )
    logger.info(f"Built {fmt_int(len(positives))} positive and {fmt_int(len(negatives))} negative pairs")

    section("Phase 1: per-scale hex2vec pretraining")

    E_h, E_w, E_s, _ = train_phase1(
        T_h, T_w, T_s,
        positives, negatives,
        dim_h    = hf.hourly_matrix.shape[1],
        dim_w    = hf.weekly_matrix.shape[1],
        dim_s    = hf.static_matrix.shape[1],
        device   = device,
        emb_dim_h = cfg.phase1.emb_dim_h,
        emb_dim_w = cfg.phase1.emb_dim_w,
        emb_dim_s = cfg.phase1.emb_dim_s,
        n_epochs  = cfg.phase1.n_epochs,
        batch_size = cfg.phase1.batch_size,
        lr         = cfg.phase1.lr,
        figures_dir = cfg.paths.figures_dir,
    )

    section("Phase 2: multi-scale fusion")
    final_emb, _, alpha = train_phase2(
        E_h, E_w, E_s,
        device    = device,
        emb_dim_h = cfg.phase1.emb_dim_h,
        emb_dim_w = cfg.phase1.emb_dim_w,
        emb_dim_s = cfg.phase1.emb_dim_s,
        latent_dim = cfg.phase2.latent_dim,
        final_dim  = cfg.phase2.final_dim,
        n_epochs   = cfg.phase2.n_epochs,
        batch_size = cfg.phase2.batch_size,
        lr         = cfg.phase2.lr,
        figures_dir = cfg.paths.figures_dir,
    )

    section("Done")
    logger.info(f"Final embedding shape: {final_emb.shape[0]:,} x {final_emb.shape[1]}")

    out = save_embedding_artifacts(cfg, hf, E_h, E_w, E_s, final_emb, alpha)
    logger.info(f"Artifacts saved to {out}/")


if __name__ == "__main__":
    main()
