"""Shared boilerplate for loading/saving pipeline artifacts (data, hex features, embeddings)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data.loader import load as load_crime_data
from src.features.hex_features import HexFeatures, build_hex_features


def load_raw_data(cfg: ExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    return load_crime_data(cfg.paths.data_path, verbose=verbose)


def load_hex_features(cfg: ExperimentConfig, df: pd.DataFrame = None) -> HexFeatures:
    if df is None:
        df = load_raw_data(cfg, verbose=False)
    return build_hex_features(df, cfg.features.resolution, cfg.features.min_events)


def load_embedding_artifacts(cfg: ExperimentConfig):
    """Load final_emb.npy, hex_ids.json, and metadata.json from a previous run."""
    emb_dir = Path(cfg.paths.embeddings_dir)

    final_emb = np.load(emb_dir / "final_emb.npy")
    with open(emb_dir / "hex_ids.json") as f:
        hex_ids = json.load(f)
    with open(emb_dir / "metadata.json") as f:
        metadata = json.load(f)

    return final_emb, hex_ids, metadata


def save_embedding_artifacts(cfg: ExperimentConfig, hf: HexFeatures,
                             E_h, E_w, E_s, final_emb: np.ndarray, alpha: float) -> Path:
    """Save the trained embeddings and their metadata."""
    out = Path(cfg.paths.embeddings_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "final_emb.npy", final_emb)
    np.save(out / "E_h.npy", E_h.cpu().numpy())
    np.save(out / "E_w.npy", E_w.cpu().numpy())
    np.save(out / "E_s.npy", E_s.cpu().numpy())

    with open(out / "hex_ids.json", "w") as f:
        json.dump(hf.hex_ids, f)

    with open(out / "metadata.json", "w") as f:
        json.dump({
            "hex_crime_count": hf.hex_crime_count,
            "hex_to_alcaldia": hf.hex_to_alcaldia,
            "alpha":           alpha,
            "resolution":      cfg.features.resolution,
            "min_events":      cfg.features.min_events,
            "N":               hf.N,
            "C":               hf.C,
        }, f, indent=2)

    return out
