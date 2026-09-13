"""Standalone hyperparameter search for Phase1/Phase2 and for the B3/B4
baselines (separate from the main pipeline).

Trains a reduced embedding per trial (the epoch count is itself one of the
tuned hyperparameters, within [min_epochs, max_epochs]) and scores it by
mean silhouette across cfg.eval.k_values, searching with Optuna's TPE
sampler. Reproducible: the sampler and every trial's torch/numpy seeds are
fixed via cfg.tuning.seed. k_ring and n_neg are kept fixed, not tuned.

The proposed model is tuned first. B3 (Autoencoder) and B4 (Hex2Vec) - the
two baselines in evaluate.py with real capacity hyperparameters - are then
tuned with the same objective, trial budget, and epoch range, and with
target_dim fixed to the tuned model's final_dim, so the comparison reported
in evaluate.py is against baselines given an equal chance, not against
arbitrary defaults. PCA and NMF are not tuned: once target_dim is fixed,
neither has a comparable capacity hyperparameter to search over.

This step ONLY searches for and reports the best hyperparameters - it does
not touch src/outputs/embeddings and it is not part of run_all.py. Running
the main pipeline / evaluate with the resulting values is a separate, later
step.

Usage:
    python -m src.pipeline.tune
    python -m src.pipeline.tune --n-trials 100 --min-epochs 20 --max-epochs 150
"""

import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import optuna
from optuna.samplers import TPESampler

from src.config import ExperimentConfig
from src.common.io import load_raw_data, load_hex_features
from src.common.log import get_logger, section, print_table
from src.features.hex_features import to_tensors
from src.models.phase1 import build_pairs
from src.evaluation.report import build_input_matrix
from src.evaluation.metrics import suggest_k_values
from src.tuning.objective import make_objective, make_autoencoder_objective, make_hex2vec_objective

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for optimal hyperparameters for the model and the B3/B4 baselines.")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of trials per study (default: cfg.tuning.n_trials).")
    parser.add_argument("--min-epochs", type=int, default=None,
                        help="Lower bound of the tuned epoch range (default: cfg.tuning.min_epochs).")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Upper bound of the tuned epoch range (default: cfg.tuning.max_epochs).")
    return parser.parse_args()


def _run_study(name: str, objective, n_trials: int, seed: int) -> optuna.Study:
    section(f"Tuning: {name}")
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=name)

    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start

    logger.info(f"Finished {len(study.trials)} trials in {elapsed:.1f}s. "
                f"Best mean silhouette: {study.best_value:.4f}")
    rows = [
        [k, f"{v:.6g}" if isinstance(v, float) else str(v)]
        for k, v in study.best_params.items()
    ]
    print_table(["Hyperparameter", "Best value"], rows, aligns=["l", "r"])
    return study


def main():
    args = parse_args()
    cfg  = ExperimentConfig()
    if args.n_trials is not None:
        cfg.tuning.n_trials = args.n_trials
    if args.min_epochs is not None:
        cfg.tuning.min_epochs = args.min_epochs
    if args.max_epochs is not None:
        cfg.tuning.max_epochs = args.max_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    section("Hyperparameter search (standalone, does not affect the main pipeline)")
    logger.info(f"Device: {device}")
    logger.info(f"Trials per study: {cfg.tuning.n_trials}, "
                f"epochs range: [{cfg.tuning.min_epochs}, {cfg.tuning.max_epochs}] "
                f"(step {cfg.tuning.epoch_step}), seed: {cfg.tuning.seed}")

    section("Loading data and building hex features")
    df = load_raw_data(cfg)
    hf = load_hex_features(cfg, df=df)
    T_h, T_w, T_s = to_tensors(hf, device)

    cfg.eval.k_values = suggest_k_values(hf.N)
    logger.info(f"K sweep for the silhouette objective (centered on sqrt(N/2)≈{(hf.N / 2) ** 0.5:.1f} "
                f"for N={hf.N:,} hexagons): {cfg.eval.k_values}")

    # 1) Proposed model
    model_objective = make_objective(cfg, hf, T_h, T_w, T_s, device)
    model_study = _run_study("proposed model", model_objective, cfg.tuning.n_trials, cfg.tuning.seed)
    target_dim = model_study.best_params["final_dim"]

    # 2) B3 / B4 baselines, tuned with the same objective/budget and the same
    # target_dim as the proposed model, so evaluate.py's comparison is fair.
    X_raw, _ = build_input_matrix(hf.hourly_matrix, hf.weekly_matrix, hf.static_matrix)
    positives, negatives = build_pairs(hf.hex_ids, hf.hex_to_idx)

    ae_objective = make_autoencoder_objective(cfg, X_raw, target_dim, device)
    ae_study = _run_study("B3: Autoencoder", ae_objective, cfg.tuning.n_trials, cfg.tuning.seed)

    hex2vec_objective = make_hex2vec_objective(cfg, positives, negatives, X_raw, target_dim, device)
    hex2vec_study = _run_study("B4: Hex2Vec", hex2vec_objective, cfg.tuning.n_trials, cfg.tuning.seed)

    section("Done")
    out_dir = cfg.tuning.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_params.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_trials":    cfg.tuning.n_trials,
            "epoch_range": [cfg.tuning.min_epochs, cfg.tuning.max_epochs],
            "seed":        cfg.tuning.seed,
            "target_dim":  target_dim,
            "model": {
                "best_value":  model_study.best_value,
                "best_params": model_study.best_params,
            },
            "baseline_autoencoder": {
                "best_value":  ae_study.best_value,
                "best_params": ae_study.best_params,
            },
            "baseline_hex2vec": {
                "best_value":  hex2vec_study.best_value,
                "best_params": hex2vec_study.best_params,
            },
        }, f, indent=2)

    logger.info(f"Best hyperparameters (model + B3 + B4) saved to {out_path}")
    logger.info("Next step (separate, later): run the main pipeline / evaluate using these values.")


if __name__ == "__main__":
    main()
