"""Single source of truth for paths and hyperparameters used across the pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    data_path: Path = PROJECT_ROOT / "data" / "carpetasFGJ_pipeline.csv"
    embeddings_dir: Path = PROJECT_ROOT / "src" / "outputs" / "embeddings"
    figures_dir: Path = PROJECT_ROOT / "src" / "outputs" / "figures"


@dataclass
class FeatureConfig:
    resolution: int = 8
    min_events: int = 30


@dataclass
class Phase1Config:
    emb_dim_h: int = 32
    emb_dim_w: int = 8
    emb_dim_s: int = 64
    hidden_dim: int = 32
    batch_size: int = 64
    n_epochs: int = 20
    lr: float = 0.0006626250104022493 
    k_ring: int = 1
    n_neg: int = 3


@dataclass
class Phase2Config:
    latent_dim: int = 8
    final_dim: int = 32
    n_epochs: int = 50
    batch_size: int = 256
    lr: float = 0.002075640733128536


@dataclass
class EvalConfig:
    # Static fallback only. In practice this is overwritten by
    # evaluation.metrics.suggest_k_values(hf.N), which derives the sweep
    # from the actual number of hexagons instead of a fixed guess.
    k_values: list = field(default_factory=lambda: [5, 10, 15, 20])
    random_state: int = 42
    # B3: Autoencoder. 
    ae_epochs: int = 20
    ae_lr: float = 0.00010216619941826734
    ae_batch: int = 256
    # B4: Hex2Vec. 
    hex2vec_epochs: int = 90
    hex2vec_lr: float = 0.00012657107246919839
    hex2vec_hidden_dim: int = 32
    names: dict = field(default_factory=lambda: {
        "pca":     "B1: PCA",
        "nmf":     "B2: NMF",
        "ae":      "B3: Autoencoder",
        "hex2vec": "B4: Hex2Vec",
        "modelo":  "M:  Proposed Model",
    })


@dataclass
class AnomalyConfig:
    k_values_sensitivity: list = field(default_factory=lambda: [10, 16, 22, 32, 45])
    k_value: int = 32
    top_percentile: float = 0.05


@dataclass
class TuningConfig:
    """Settings for the standalone hyperparameter search (src/pipeline/tune.py).

    Runs Phase1 + Phase2 per trial (n_epochs is itself tuned within
    [min_epochs, max_epochs]) and scores each trial by mean silhouette over
    eval.k_values. Independent from the main pipeline: it never touches
    src/outputs/embeddings. k_ring and n_neg are kept fixed at their
    Phase1Config defaults, not tuned.
    """
    n_trials: int = 40
    min_epochs: int = 20
    max_epochs: int = 100
    epoch_step: int = 10
    seed: int = 42
    study_name: str = "hex_embeddings_tuning"
    results_dir: Path = PROJECT_ROOT / "src" / "outputs" / "tuning"


@dataclass
class ExperimentConfig:
    seed: int = 32
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    eval: EvalConfig = field(default_factory=EvalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
