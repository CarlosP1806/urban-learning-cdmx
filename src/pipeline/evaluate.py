import random
import numpy as np
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from src.config import ExperimentConfig
from src.common.io import load_hex_features, load_embedding_artifacts
from src.common.log import get_logger, section
from src.models.phase1 import build_pairs
from src.models.baselines import run_pca, run_nmf, run_autoencoder, train_hex2vec
from src.evaluation.report import build_input_matrix, print_final_table, plot_per_k
from src.evaluation.metrics import evaluate_embedding

logger = get_logger(__name__)


def main():
    cfg    = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(cfg.eval.random_state)
    np.random.seed(cfg.eval.random_state)
    torch.manual_seed(cfg.eval.random_state)

    final_emb, hex_ids, meta = load_embedding_artifacts(cfg)
    hf = load_hex_features(cfg)

    X_raw, X_nonneg = build_input_matrix(
        hf.hourly_matrix, hf.weekly_matrix, hf.static_matrix
    )

    target_dim = final_emb.shape[1]

    positives, negatives = build_pairs(hf.hex_ids, hf.hex_to_idx)

    section("Training baseline embeddings")
    embeddings = {
        "pca":     run_pca(X_raw, target_dim),
        "nmf":     run_nmf(X_nonneg, target_dim),
        "ae":      run_autoencoder(X_raw, target_dim, device),
        "hex2vec": train_hex2vec(positives, negatives, X_raw, target_dim, device),
        "modelo":  final_emb.copy(),
    }
    logger.info(f"Built {len(embeddings)} embeddings of dimension {target_dim}")

    section("Scoring embeddings")
    results = [
        evaluate_embedding(name, emb, cfg.eval.names, cfg.eval.k_values)
        for name, emb in tqdm(embeddings.items(), desc="Clustering metrics", unit="method")
    ]

    print_final_table(results, cfg.eval.k_values)
    plot_per_k(embeddings, cfg.eval.names, cfg.eval.k_values,
              output_dir=cfg.paths.figures_dir / "evaluation")


if __name__ == "__main__":
    main()
