import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.config import Phase1Config, PathsConfig
from src.common.viz import plot_training_loss
from src.common.log import get_logger

logger = get_logger(__name__)

_DEFAULTS = Phase1Config()

EMB_DIM_H  = _DEFAULTS.emb_dim_h
EMB_DIM_W  = _DEFAULTS.emb_dim_w
EMB_DIM_S  = _DEFAULTS.emb_dim_s
HIDDEN_DIM = _DEFAULTS.hidden_dim
BATCH_SIZE = _DEFAULTS.batch_size
N_EPOCHS   = _DEFAULTS.n_epochs
LR         = _DEFAULTS.lr


class Hex2VecDataset(Dataset):
    """Geographic pairs (positive and negative) for hex2vec training."""

    def __init__(self, positives, negatives, T_h, T_w, T_s):
        self.pairs = (
            [(i, j, 1.0) for i, j in positives] +
            [(i, j, 0.0) for i, j in negatives]
        )
        self.T_h = T_h
        self.T_w = T_w
        self.T_s = T_s

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        return (
            self.T_h[i], self.T_w[i], self.T_s[i],
            self.T_h[j], self.T_w[j], self.T_s[j],
            torch.tensor(label, dtype=torch.float32),
        )


class ScaleEmbedder(nn.Module):
    """MLP that maps a raw profile to an embedding of dimension emb_dim."""

    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class Phase1Model(nn.Module):
    """
    Pure hex2vec: learns per-scale embeddings using only the
    geographic signal (BCE over positive/negative H3 pairs).
    """

    def __init__(self, dim_h, dim_w, dim_s,
                 emb_dim_h=EMB_DIM_H,
                 emb_dim_w=EMB_DIM_W,
                 emb_dim_s=EMB_DIM_S,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.emb_h = ScaleEmbedder(dim_h, emb_dim_h, hidden_dim)
        self.emb_w = ScaleEmbedder(dim_w, emb_dim_w, hidden_dim)
        self.emb_s = ScaleEmbedder(dim_s, emb_dim_s, hidden_dim)

    def _score(self, embedder, x_t, x_c):
        return torch.sigmoid((embedder(x_t) * embedder(x_c)).sum(dim=1))

    def forward(self, h_t, w_t, s_t, h_c, w_c, s_c):
        return (
            self._score(self.emb_h, h_t, h_c),
            self._score(self.emb_w, w_t, w_c),
            self._score(self.emb_s, s_t, s_c),
        )

    def get_embeddings(self, T_h, T_w, T_s):
        with torch.no_grad():
            return self.emb_h(T_h), self.emb_w(T_w), self.emb_s(T_s)


def build_pairs(hex_ids: list, hex_to_idx: dict,
                k_ring: int = _DEFAULTS.k_ring, n_neg: int = _DEFAULTS.n_neg):
    """Positive pairs (H3 ring-1 neighbors) and negative pairs (distant hexagons)."""
    import h3 as h3lib
    positives, negatives = [], []
    idx_array = np.arange(len(hex_ids))

    for hex_id in hex_ids:
        i = hex_to_idx[hex_id]
        for nbr in h3lib.grid_ring(hex_id, k_ring):
            if nbr in hex_to_idx:
                positives.append((i, hex_to_idx[nbr]))
        excluded   = {hex_to_idx[x] for x in h3lib.grid_disk(hex_id, 2)
                      if x in hex_to_idx}
        candidates = [j for j in idx_array if j not in excluded]
        if len(candidates) >= n_neg:
            for j in np.random.choice(candidates, n_neg, replace=False):
                negatives.append((i, int(j)))

    return positives, negatives


def train_phase1(T_h, T_w, T_s,
                 positives, negatives,
                 dim_h, dim_w, dim_s,
                 device,
                 emb_dim_h=EMB_DIM_H,
                 emb_dim_w=EMB_DIM_W,
                 emb_dim_s=EMB_DIM_S,
                 hidden_dim=HIDDEN_DIM,
                 n_epochs=N_EPOCHS,
                 batch_size=BATCH_SIZE,
                 lr=LR,
                 figures_dir: Path = PathsConfig().figures_dir,
                 verbose: bool = True) -> tuple:
    """
    Train Phase1Model (pure hex2vec) and return the three embeddings.

    Returns
    -------
    E_h, E_w, E_s : torch.Tensor detached on CPU
    model         : trained Phase1Model
    """
    dataset   = Hex2VecDataset(positives, negatives, T_h, T_w, T_s)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = Phase1Model(dim_h, dim_w, dim_s,
                            emb_dim_h, emb_dim_w, emb_dim_s, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    if verbose:
        logger.info(f"Phase 1: {len(dataset):,} pairs, {n_epochs} epochs, batch size {batch_size}")

    loss_history = []
    progress = tqdm(range(n_epochs), desc="Phase 1", unit="epoch", disable=not verbose)

    for epoch in progress:
        model.train()
        total_loss = 0.0

        for batch in loader:
            h_t, w_t, s_t, h_c, w_c, s_c, labels = batch

            optimizer.zero_grad()
            s_h, s_w, s_s = model(h_t, w_t, s_t, h_c, w_c, s_c)

            loss = (criterion(s_h, labels) +
                    criterion(s_w, labels) +
                    criterion(s_s, labels))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        loss_history.append(epoch_loss)
        progress.set_postfix(loss=f"{epoch_loss:.4f}")

    model.eval()
    E_h, E_w, E_s = model.get_embeddings(T_h, T_w, T_s)

    if figures_dir is not None:
        plot_training_loss(loss_history, "Training Loss - Phase 1",
                            figures_dir / "phase1_loss.png")
    if verbose:
        logger.info(f"Phase 1 done: final loss {loss_history[-1]:.4f}")

    return E_h.detach(), E_w.detach(), E_s.detach(), model
