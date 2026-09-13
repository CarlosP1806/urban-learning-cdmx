"""Optuna objectives: train a reduced-epoch embedding for one trial's
hyperparameters and score it by mean silhouette across eval.k_values.

Includes objectives for the proposed model (Phase1+Phase2) and for the B3
(Autoencoder) / B4 (Hex2Vec) baselines, so both get the same tuning
treatment - same objective, same trial budget, same epoch range - instead
of comparing a tuned model against baselines left at arbitrary defaults.

Hex features (and, for the baselines, the raw feature matrix / geographic
pairs) are built once outside the objective and reused across all trials,
which is what makes the search fast enough to run standalone.
"""

import numpy as np
import torch

from src.config import ExperimentConfig
from src.features.hex_features import HexFeatures
from src.models.phase1 import build_pairs, train_phase1
from src.models.phase2 import train_phase2
from src.models.baselines import run_autoencoder, train_hex2vec
from src.evaluation.metrics import cluster_labels, metric_silhouette
from src.tuning.search_space import (suggest_phase1_config, suggest_phase2_config,
                                     suggest_autoencoder_params, suggest_hex2vec_params)


def _mean_silhouette(emb, k_values) -> float:
    scores = [metric_silhouette(emb, cluster_labels(emb, k)) for k in k_values]
    return float(np.mean(scores))


def make_objective(cfg: ExperimentConfig, hf: HexFeatures, T_h, T_w, T_s, device):
    """Build an Optuna objective closure that reuses precomputed hex features."""

    def objective(trial) -> float:
        torch.manual_seed(cfg.tuning.seed)
        np.random.seed(cfg.tuning.seed)

        p1 = suggest_phase1_config(trial, cfg.tuning)
        p2 = suggest_phase2_config(trial, cfg.tuning)

        positives, negatives = build_pairs(
            hf.hex_ids, hf.hex_to_idx, k_ring=p1.k_ring, n_neg=p1.n_neg,
        )

        E_h, E_w, E_s, _ = train_phase1(
            T_h, T_w, T_s, positives, negatives,
            dim_h=hf.hourly_matrix.shape[1],
            dim_w=hf.weekly_matrix.shape[1],
            dim_s=hf.static_matrix.shape[1],
            device=device,
            emb_dim_h=p1.emb_dim_h, emb_dim_w=p1.emb_dim_w, emb_dim_s=p1.emb_dim_s,
            hidden_dim=p1.hidden_dim,
            n_epochs=p1.n_epochs, batch_size=p1.batch_size, lr=p1.lr,
            figures_dir=None, verbose=False,
        )

        final_emb, _, _ = train_phase2(
            E_h, E_w, E_s,
            device=device,
            emb_dim_h=p1.emb_dim_h, emb_dim_w=p1.emb_dim_w, emb_dim_s=p1.emb_dim_s,
            latent_dim=p2.latent_dim, final_dim=p2.final_dim,
            n_epochs=p2.n_epochs, batch_size=p2.batch_size, lr=p2.lr,
            figures_dir=None, verbose=False,
        )

        return _mean_silhouette(final_emb, cfg.eval.k_values)

    return objective


def make_autoencoder_objective(cfg: ExperimentConfig, X_raw, target_dim: int, device):
    """B3 baseline objective: same target_dim as the tuned model, for a fair comparison."""

    def objective(trial) -> float:
        torch.manual_seed(cfg.tuning.seed)
        np.random.seed(cfg.tuning.seed)

        params = suggest_autoencoder_params(trial, cfg.tuning)
        emb = run_autoencoder(
            X_raw, target_dim, device,
            n_epochs=params["n_epochs"], lr=params["lr"],
        )
        return _mean_silhouette(emb, cfg.eval.k_values)

    return objective


def make_hex2vec_objective(cfg: ExperimentConfig, positives, negatives, X_raw,
                           target_dim: int, device):
    """B4 baseline objective: same target_dim, positives/negatives as the tuned model."""

    def objective(trial) -> float:
        torch.manual_seed(cfg.tuning.seed)
        np.random.seed(cfg.tuning.seed)

        params = suggest_hex2vec_params(trial, cfg.tuning)
        emb = train_hex2vec(
            positives, negatives, X_raw, target_dim, device,
            n_epochs=params["n_epochs"], lr=params["lr"], hidden_dim=params["hidden_dim"],
        )
        return _mean_silhouette(emb, cfg.eval.k_values)

    return objective
