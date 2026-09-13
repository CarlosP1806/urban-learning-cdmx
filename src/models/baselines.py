"""Baseline embedding methods used to benchmark the proposed model in evaluation/report.py."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.decomposition import PCA, NMF

from src.config import EvalConfig

_DEFAULTS = EvalConfig()

RANDOM_STATE = _DEFAULTS.random_state
AE_EPOCHS    = _DEFAULTS.ae_epochs
AE_LR        = _DEFAULTS.ae_lr
AE_BATCH     = _DEFAULTS.ae_batch
HEX2VEC_EPOCHS     = _DEFAULTS.hex2vec_epochs
HEX2VEC_LR         = _DEFAULTS.hex2vec_lr
HEX2VEC_HIDDEN_DIM = _DEFAULTS.hex2vec_hidden_dim


def run_pca(X_raw, target_dim):
    pca  = PCA(n_components=target_dim, random_state=RANDOM_STATE)
    emb  = pca.fit_transform(X_raw)
    return emb.astype(np.float32)


def run_nmf(X_nonneg, target_dim):
    nmf = NMF(n_components=target_dim,
              init="nndsvda",
              max_iter=500,
              random_state=RANDOM_STATE)
    emb = nmf.fit_transform(X_nonneg)
    return emb.astype(np.float32)


class SimpleAutoencoder(nn.Module):
    """
    Standard dense autoencoder, with no multi-scale structure or contrastive loss.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        h = max(input_dim // 2, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, h),
            nn.ReLU(),
            nn.Linear(h, input_dim)
        )

    def forward(self, x):
        z   = self.encoder(x)
        rec = self.decoder(z)
        return z, rec


def run_autoencoder(X_raw, target_dim, device,
                    n_epochs=AE_EPOCHS, lr=AE_LR, batch_size=AE_BATCH):
    X_t      = torch.tensor(X_raw).to(device)
    dataset  = TensorDataset(X_t)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model    = SimpleAutoencoder(X_raw.shape[1], target_dim).to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=lr)

    progress = tqdm(range(n_epochs), desc="B3: Autoencoder", unit="epoch", leave=False)
    for epoch in progress:
        model.train()
        total_loss = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            _, rec = model(xb)
            loss   = F.mse_loss(rec, xb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        progress.set_postfix(loss=f"{total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        emb, _ = model(X_t)
    return emb.cpu().numpy().astype(np.float32)


## Pure Hex2Vec
class Hex2VecDataset(Dataset):
    """
    Minimal dataset: each sample is a (target, context, label) pair.
    label=1 if they are ring-1 neighbors, label=0 if negatives.

    Unlike HybridDataset from the main pipeline:
    - Does not generate augmentations
    - Works on the full concatenated profile (not per scale)
    - Does not return contrastive pairs
    """
    def __init__(self, positives, negatives, X: torch.Tensor):
        self.pairs = (
            [(i, j, 1.0) for i, j in positives] +
            [(i, j, 0.0) for i, j in negatives]
        )
        self.X = X

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        return (
            self.X[i],
            self.X[j],
            torch.tensor(label, dtype=torch.float32)
        )


class Hex2VecModel(nn.Module):
    """
    Skip-gram encoder over the concatenated crime profile.
    """
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, emb_dim)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x_target: torch.Tensor,
                x_context: torch.Tensor) -> torch.Tensor:
        """
        Returns similarity score in (0,1) for each pair.
        Dot product and sigmoid, same as in the main pipeline.
        """
        e_t = self.embed(x_target)
        e_c = self.embed(x_context)
        return torch.sigmoid((e_t * e_c).sum(dim=1))


def train_hex2vec(
    positives: list,
    negatives: list,
    X_raw: np.ndarray,
    target_dim: int,
    device: torch.device,
    n_epochs: int = HEX2VEC_EPOCHS,
    batch_size: int = 256,
    lr: float = HEX2VEC_LR,
    hidden_dim: int = HEX2VEC_HIDDEN_DIM,
) -> np.ndarray:
    """
    Train pure hex2vec and return the learned embeddings (N, target_dim).
    """
    input_dim = X_raw.shape[1]
    X_tensor  = torch.tensor(X_raw, dtype=torch.float32).to(device)

    dataset   = Hex2VecDataset(positives, negatives, X_tensor)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, drop_last=False)

    model     = Hex2VecModel(input_dim, target_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    progress = tqdm(range(n_epochs), desc="B4: Hex2Vec", unit="epoch", leave=False)
    for epoch in progress:
        model.train()
        total_loss = 0.0
        for x_t, x_c, labels in loader:
            optimizer.zero_grad()
            scores = model(x_t, x_c)
            loss   = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        progress.set_postfix(loss=f"{total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        emb = model.embed(X_tensor)

    return emb.cpu().numpy().astype(np.float32)
