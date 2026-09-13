"""Search space for the Optuna hyperparameter search over Phase1/Phase2 configs,
and over the B3 (Autoencoder) / B4 (Hex2Vec) baselines used in evaluate.py.

Only model/training hyperparameters are tuned here. Feature engineering
(H3 resolution, min_events) and the evaluation k_values are kept fixed so
that every trial is scored on the same set of hexagons and the same
clustering grid, isolating the effect of the hyperparameters themselves.

B3/B4 are tuned with the same epoch range, trial budget, and objective
(mean silhouette) as the proposed model - and the same target_dim, fixed to
the tuned model's final_dim - so the comparison in the paper is against
baselines given an equal chance, not against arbitrary defaults. PCA and
NMF are not tuned: once target_dim is fixed, they have no comparable
capacity hyperparameter to search over.
"""

from src.config import Phase1Config, Phase2Config, TuningConfig

PHASE1_EMB_DIM_H_CHOICES = [8, 16, 32]
PHASE1_EMB_DIM_W_CHOICES = [8, 16]
PHASE1_EMB_DIM_S_CHOICES = [16, 32, 64]
PHASE1_HIDDEN_DIM_CHOICES = [32, 64, 128]
PHASE1_LR_RANGE = (1e-4, 1e-2)

PHASE2_LATENT_DIM_CHOICES = [8, 16]
PHASE2_FINAL_DIM_CHOICES = [16, 32, 64]
PHASE2_LR_RANGE = (1e-4, 1e-2)

# Same ranges as Phase1/Phase2 above, reused for the baselines so all methods
# search lr/hidden_dim over an identical grid.
BASELINE_LR_RANGE = (1e-4, 1e-2)
BASELINE_HIDDEN_DIM_CHOICES = PHASE1_HIDDEN_DIM_CHOICES

_PHASE1_DEFAULTS = Phase1Config()


def suggest_phase1_config(trial, tuning: TuningConfig) -> Phase1Config:
    return Phase1Config(
        emb_dim_h=trial.suggest_categorical("emb_dim_h", PHASE1_EMB_DIM_H_CHOICES),
        emb_dim_w=trial.suggest_categorical("emb_dim_w", PHASE1_EMB_DIM_W_CHOICES),
        emb_dim_s=trial.suggest_categorical("emb_dim_s", PHASE1_EMB_DIM_S_CHOICES),
        hidden_dim=trial.suggest_categorical("hidden_dim", PHASE1_HIDDEN_DIM_CHOICES),
        lr=trial.suggest_float("phase1_lr", *PHASE1_LR_RANGE, log=True),
        n_epochs=trial.suggest_int("phase1_epochs", tuning.min_epochs, tuning.max_epochs,
                                   step=tuning.epoch_step),
        k_ring=_PHASE1_DEFAULTS.k_ring,
        n_neg=_PHASE1_DEFAULTS.n_neg,
    )


def suggest_phase2_config(trial, tuning: TuningConfig) -> Phase2Config:
    return Phase2Config(
        latent_dim=trial.suggest_categorical("latent_dim", PHASE2_LATENT_DIM_CHOICES),
        final_dim=trial.suggest_categorical("final_dim", PHASE2_FINAL_DIM_CHOICES),
        lr=trial.suggest_float("phase2_lr", *PHASE2_LR_RANGE, log=True),
        n_epochs=trial.suggest_int("phase2_epochs", tuning.min_epochs, tuning.max_epochs,
                                   step=tuning.epoch_step),
    )


def suggest_autoencoder_params(trial, tuning: TuningConfig) -> dict:
    """B3 baseline (SimpleAutoencoder): tunes lr and n_epochs only - its
    hidden layer width is already derived from target_dim/input_dim inside
    run_autoencoder, so there is no separate capacity knob to search."""
    return dict(
        lr=trial.suggest_float("ae_lr", *BASELINE_LR_RANGE, log=True),
        n_epochs=trial.suggest_int("ae_epochs", tuning.min_epochs, tuning.max_epochs,
                                   step=tuning.epoch_step),
    )


def suggest_hex2vec_params(trial, tuning: TuningConfig) -> dict:
    """B4 baseline (Hex2VecModel): tunes hidden_dim, lr and n_epochs, mirroring
    Phase1's search space so both encoders are searched with equal granularity."""
    return dict(
        hidden_dim=trial.suggest_categorical("hex2vec_hidden_dim", BASELINE_HIDDEN_DIM_CHOICES),
        lr=trial.suggest_float("hex2vec_lr", *BASELINE_LR_RANGE, log=True),
        n_epochs=trial.suggest_int("hex2vec_epochs", tuning.min_epochs, tuning.max_epochs,
                                   step=tuning.epoch_step),
    )
