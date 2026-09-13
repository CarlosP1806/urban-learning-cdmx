import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.config import Phase2Config, PathsConfig
from src.common.viz import plot_training_loss
from src.common.log import get_logger

logger = get_logger(__name__)

_DEFAULTS = Phase2Config()

LATENT_DIM = _DEFAULTS.latent_dim
FINAL_DIM  = _DEFAULTS.final_dim
N_EPOCHS   = _DEFAULTS.n_epochs
BATCH_SIZE = _DEFAULTS.batch_size
LR         = _DEFAULTS.lr


class ScaleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        lat = self.encoder(x)
        rec = self.decoder(lat)
        return lat, rec


class LogAttentionFusion(nn.Module):
    def __init__(self, latent_dim, final_dim):
        super().__init__()
        self.W_Q   = nn.Linear(latent_dim, latent_dim)
        self.W_K   = nn.Linear(latent_dim, latent_dim)
        self.W_V   = nn.Linear(latent_dim, latent_dim)
        self.out   = nn.Linear(latent_dim, final_dim)
        self.scale = latent_dim ** 0.5

    def forward(self, e_h, e_w, e_s):
        R   = torch.stack([e_h, e_w, e_s], dim=1)
        Q   = self.W_Q(R)
        K   = self.W_K(R)
        V   = self.W_V(R)
        A   = F.log_softmax(
                torch.bmm(Q, K.transpose(1,2)) / self.scale,
                dim=-1
              )
        out = torch.bmm(torch.exp(A), V).mean(dim=1)
        return self.out(out)


class Phase2Model(nn.Module):
    def __init__(self, emb_dim_h, emb_dim_w, emb_dim_s, latent_dim=LATENT_DIM, final_dim=FINAL_DIM):
        super().__init__()
        self.ae_h   = ScaleAutoencoder(emb_dim_h, latent_dim)
        self.ae_w   = ScaleAutoencoder(emb_dim_w, latent_dim)
        self.ae_s   = ScaleAutoencoder(emb_dim_s, latent_dim)
        self.fusion = LogAttentionFusion(latent_dim, final_dim)

    def forward(self, e_h, e_w, e_s):
        lat_h, rec_h = self.ae_h(e_h)
        lat_w, rec_w = self.ae_w(e_w)
        lat_s, rec_s = self.ae_s(e_s)
        final        = self.fusion(lat_h, lat_w, lat_s)
        return (final,
                (lat_h, lat_w, lat_s),
                (rec_h, rec_w, rec_s))


def reconstruction_loss(originals, reconstructeds):
    return sum(F.mse_loss(r, o) for o, r in zip(originals, reconstructeds))


def contrastive_loss_scales(lat_h, lat_w, lat_s, tau=0.07):
    """NT-Xent across scales: same region, different scales = positive."""
    def nce(z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        sim = torch.mm(z_i, z_j.T) / tau
        labels = torch.arange(z_i.size(0)).to(z_i.device)
        return F.cross_entropy(sim, labels)
    return (nce(lat_h, lat_w) + nce(lat_w, lat_s) + nce(lat_s, lat_h)) / 3.0


def train_phase2(E_h, E_w, E_s,
                 device,
                 emb_dim_h, emb_dim_w, emb_dim_s,
                 latent_dim=LATENT_DIM,
                 final_dim=FINAL_DIM,
                 n_epochs=N_EPOCHS,
                 batch_size=BATCH_SIZE,
                 lr=LR,
                 figures_dir: Path = PathsConfig().figures_dir) -> tuple:
    """
    Train Phase2Model and return the final embedding.

    Returns
    -------
    final_emb : np.ndarray (N, final_dim)
    model     : trained Phase2Model
    alpha     : learned value of alpha
    """
    N      = E_h.shape[0]
    model  = Phase2Model(emb_dim_h, emb_dim_w, emb_dim_s,
                         latent_dim, final_dim).to(device)
    ALPHA  = nn.Parameter(torch.tensor(0.5))
    opt    = torch.optim.Adam(list(model.parameters()) + [ALPHA], lr=lr)

    n_batches = max(1, N // batch_size)

    logger.info(f"Phase 2: {N:,} hexagons, {n_epochs} epochs, {n_batches} batches/epoch")

    loss_history = []
    progress = tqdm(range(n_epochs), desc="Phase 2", unit="epoch")

    for epoch in progress:
        model.train()
        ALPHA.data.clamp_(0.0, 1.0)
        perm       = torch.randperm(N)
        total_loss = 0.0

        for i in range(n_batches):
            idx   = perm[i * batch_size:(i+1) * batch_size]
            e_h_b = E_h[idx].to(device)
            e_w_b = E_w[idx].to(device)
            e_s_b = E_s[idx].to(device)

            opt.zero_grad()
            final, (lat_h, lat_w, lat_s), (rec_h, rec_w, rec_s) = \
                model(e_h_b, e_w_b, e_s_b)

            alpha = torch.sigmoid(ALPHA)
            loss  = (alpha * reconstruction_loss(
                        [e_h_b, e_w_b, e_s_b], [rec_h, rec_w, rec_s]) +
                     (1 - alpha) * contrastive_loss_scales(lat_h, lat_w, lat_s))
            loss.backward()
            opt.step()
            total_loss += loss.item()

        epoch_loss = total_loss / n_batches
        loss_history.append(epoch_loss)
        progress.set_postfix(loss=f"{epoch_loss:.4f}", alpha=f"{torch.sigmoid(ALPHA).item():.3f}")

    model.eval()
    with torch.no_grad():
        final_emb, _, _ = model(E_h.to(device), E_w.to(device), E_s.to(device))

    plot_training_loss(loss_history, "Training Loss - Phase 2",
                        figures_dir / "phase2_loss.png")
    alpha_final = torch.sigmoid(ALPHA).item()
    logger.info(f"Phase 2 done: final loss {loss_history[-1]:.4f}, alpha={alpha_final:.3f}")

    return final_emb.cpu().numpy(), model, alpha_final
