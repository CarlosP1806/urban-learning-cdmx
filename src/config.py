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
    emb_dim_h: int = 16
    emb_dim_w: int = 8
    emb_dim_s: int = 32
    hidden_dim: int = 64
    batch_size: int = 256
    n_epochs: int = 175
    lr: float = 1e-3
    k_ring: int = 1
    n_neg: int = 3


@dataclass
class Phase2Config:
    latent_dim: int = 32
    final_dim: int = 64
    n_epochs: int = 175
    batch_size: int = 256
    lr: float = 1e-3


@dataclass
class EvalConfig:
    k_values: list = field(default_factory=lambda: [5, 10, 15, 20])
    random_state: int = 42
    ae_epochs: int = 60
    ae_lr: float = 1e-3
    ae_batch: int = 256
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
class ExperimentConfig:
    seed: int = 32
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    eval: EvalConfig = field(default_factory=EvalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
