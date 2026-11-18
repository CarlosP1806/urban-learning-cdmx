# Urban Representation Learning for Identifying Potential Disparities in Crime Reporting

Code, trained embeddings and supplementary figures for the paper of the same name.

The pipeline learns a 32-dimensional embedding for each H3 hexagonal cell in Mexico City from reported-crime records published by the Fiscalía General de Justicia (FGJ-CDMX). Embeddings are trained with a self-supervised objective based on geographic adjacency, evaluated against four baselines by clustering quality, and then used to compute a per-cell divergence score (Neighbor Profile Divergence, NPD) that compares each cell's reporting profile with that of its nearest cells in embedding space.

The current run covers **1,058 cells** at H3 resolution 8 and **38 crime categories**.

## Contents

- [Pipeline](#pipeline)
- [Implementation details](#implementation-details)
- [Baselines and hyperparameter search](#baselines-and-hyperparameter-search)
- [Results](#results)
- [Full clustering results](#full-clustering-results)
- [Seeds and variability](#seeds-and-variability)
- [NPD definition](#npd-definition)
- [Peers versus geographic neighbours](#peers-versus-geographic-neighbours)
- [Supplementary figures](#supplementary-figures)
- [Maps](#maps)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data](#data)
- [Interpretation](#interpretation)
- [Limitations](#limitations)
- [Planned analyses](#planned-analyses)
- [Citation](#citation)

## Pipeline

![Framework overview](figs/urban-analytics-learning-framework.png)
*Overview of the proposed framework. Incident-level records with geographic coordinates, occurrence time, and crime category are cleaned, assigned to H3 cells, and aggregated into region-level profiles. Building on these inputs, a self-supervised model learns latent representations of urban regions and uses them to identify reporting patterns that deviate from those of regions with similar learned representations. The framework comprises three stages: (i) construction of complementary regional reporting profiles, (ii) self-supervised learning and fusion of scale-specific representations, and (iii) retrieval of functional peers followed by Neighbor Profile Divergence (NPD) analysis. Steps 1–4 and 6 below implement these stages; steps 5, 7 and 8 cover evaluation, maps and tuning.*

1. **Loading** (`src/data/loader.py`, `src/data/taxonomy.py`). Reads the FGJ CSV, casts dates and coordinates, derives hour-of-day and day-of-week, and maps the free-text `delito` field to 38 standard categories through `CRIME_TAXONOMY`. Rows without coordinates, timestamp, or a mapped category are dropped. On the current file: 2,098,743 records loaded; 101,899 dropped for missing location or date; 133,917 dropped for unmapped offence types; 1,862,927 remain.

2. **Features** (`src/features/hex_features.py`). Each event is assigned to an H3 cell at resolution 8 (about 0.74 km²). Cells with fewer than 30 events are discarded: 1,179 cells have at least one event, 121 are removed, 1,058 are kept. For each remaining cell, three feature matrices are built:

   | Matrix | Shape | Contents |
   |---|---|---|
   | `hourly_matrix` | N × 24 | hour-of-day distribution |
   | `weekly_matrix` | N × 10 | day-of-week distribution; log weekday/weekend ratio; day-of-week entropy; weekend share |
   | `static_matrix` | N × 45 | crime-type proportions (38); violent share; type entropy; hour entropy; severe-crime share; median reporting delay; same-day report fraction; yearly trend slope |

   All entries are proportions or derived statistics. Event volume is not an input feature. Each matrix is standardised independently before training.

3. **Phase 1** (`src/models/phase1.py`). Three MLP encoders, one per matrix, produce embeddings of dimension 32, 8 and 64. Training pairs are geographic: ring-1 H3 neighbours are positives, three cells sampled outside radius 2 are negatives. Each encoder is trained with binary cross-entropy on the sigmoid of the pair dot product; the three losses are summed.

4. **Phase 2** (`src/models/phase2.py`). Each Phase 1 embedding passes through an autoencoder with an 8-dimensional latent. The three latents are fused by a self-attention layer into the final 32-dimensional embedding. The loss is `α · reconstruction + (1 − α) · NT-Xent`, where the contrastive positives are the same cell seen through different views and α is a learned parameter. In the current run α converged to 0.708.

5. **Evaluation** (`src/pipeline/evaluate.py`, `src/evaluation/`). KMeans is run on the final embedding and on four baselines (PCA, NMF, dense autoencoder, Hex2Vec over the concatenated profile) for a set of K values centred on √(N/2). Silhouette, Calinski–Harabasz and Davies–Bouldin are reported for each.

6. **Anomaly detection** (`src/pipeline/detect_anomalies.py`, `src/anomaly/`). For each cell, its k = 32 nearest cells by cosine distance in the final embedding are retrieved. The expected profile is the similarity-weighted mean of their profiles, and NPD is `1 − ⟨profile, expected profile⟩`. The top 5% of cells by NPD are flagged as candidates. A sensitivity sweep over k and a comparison against the observed/expected count ratio are also produced.

7. **Maps** (`src/pipeline/make_maps.py`). Maps of the processing steps and results, listed under [Maps](#maps).

8. **Tuning** (`src/pipeline/tune.py`, `src/tuning/`). Optuna search over Phase 1 and Phase 2 hyperparameters, scored by mean silhouette. Independent of the main pipeline; writes only to `src/outputs/tuning/`.

The terms "scale" and "multi-scale" in the code (`ScaleEmbedder`, `contrastive_loss_scales`) refer to the three feature views above, not to multiple spatial resolutions. A single H3 resolution and a single neighbourhood radius are used throughout.

## Implementation details

Values below are taken from `src/config.py` and the model code. There is no early stopping, no learning-rate schedule and no weight decay in any stage; all layers use PyTorch's default initialisation; Adam is the optimiser everywhere.

**Inputs.** The three matrices are standardised column-wise with `StandardScaler` (fit on all 1,058 cells) before entering Phase 1.

**Phase 1 encoders** (`ScaleEmbedder`, one per view, trained jointly):

| | Hourly | Weekly | Static |
|---|---|---|---|
| Input dimension | 24 | 10 | 45 |
| Architecture | Linear(d, 32) → ReLU → Dropout(0.2) → Linear(32, 16) → ReLU → Linear(16, e) | same | same |
| Output dimension e | 32 | 8 | 64 |

- Hidden width `hidden_dim` = 32 (see the note on `best_params.json` below).
- Pair score: sigmoid of the dot product of the two embeddings; loss `BCELoss`, summed over the three views.
- Pairs: 5,702 positives (ring-1 neighbours present in the cell set) and 3,174 negatives (3 per cell, sampled uniformly from cells outside `grid_disk(cell, 2)`, drawn once before training). Batch size 64 (139 batches per epoch), 20 epochs, learning rate 6.626250e-4.
- Dropout is active during training; the stored `E_h`, `E_w`, `E_s` are the eval-mode outputs.

**Phase 2** (`ScaleAutoencoder` × 3 and `LogAttentionFusion`):

| Autoencoder | Encoder | Decoder |
|---|---|---|
| `ae_h` (32-d input) | Linear(32, 16) → ReLU → Linear(16, 8) | Linear(8, 16) → ReLU → Linear(16, 32) |
| `ae_w` (8-d input) | Linear(8, 4) → ReLU → Linear(4, 8) | Linear(8, 4) → ReLU → Linear(4, 8) |
| `ae_s` (64-d input) | Linear(64, 32) → ReLU → Linear(32, 8) | Linear(8, 32) → ReLU → Linear(32, 64) |

- Fusion: the three 8-d latents form a sequence of 3 tokens; single-head attention with Linear(8, 8) query/key/value projections, scores divided by √8, `log_softmax` then `exp`, attention applied to the values, mean over the 3 tokens, then Linear(8, 32). No dropout in Phase 2.
- Loss: `α · Σ_views MSE(reconstruction, input) + (1 − α) · NT-Xent`. NT-Xent uses temperature τ = 0.07, L2-normalised latents, in-batch negatives, and averages the three view pairs (h–w, w–s, s–h).
- α = sigmoid(a) where a is a learned scalar initialised at 0.5 and clamped to [0, 1] at the start of every epoch; α is therefore confined to [0.5, 0.731]. Final value α = 0.708.
- Batch size 256 with `N // 256` = 4 batches per epoch (1,024 of the 1,058 cells per epoch, under a fresh random permutation each epoch), 50 epochs, learning rate 2.075641e-3.
- The stored `final_emb.npy` is the eval-mode forward pass of all cells after training.

**`hidden_dim` in `config.py` versus `best_params.json`.** `config.py` sets `phase1.hidden_dim = 32` while `src/outputs/tuning/best_params.json` records 64 as the best trial. Retraining on CPU with `train.py`'s seeding reproduces the stored embeddings with `hidden_dim = 32` (`E_h` equal to within 8e-7; `final_emb` mean per-cell cosine 0.99995, minimum 0.99936; α = 0.70794 against the stored 0.70799) and does not with 64 (mean cosine ≈ 0). The stored embeddings were therefore trained with `hidden_dim = 32`, not with the value selected by the search. Every other Phase 1 and Phase 2 value in `config.py` matches `best_params.json`.

## Baselines and hyperparameter search

All baselines take the same 79-dimensional input: the three standardised matrices concatenated (24 + 10 + 45). NMF uses a min–max rescaling of that matrix to [0, 1]. All produce 32-dimensional embeddings (`target_dim` = the model's `final_dim`).

| Baseline | Configuration |
|---|---|
| B1: PCA | 32 components, `random_state` 42 |
| B2: NMF | 32 components, `init="nndsvda"`, `max_iter` 500, `random_state` 42 |
| B3: Autoencoder | encoder Linear(79, 64) → ReLU → Dropout(0.1) → Linear(64, 32) → ReLU → Linear(32, 32); mirrored decoder; MSE; batch 256; 20 epochs; lr 1.021662e-4 |
| B4: Hex2Vec | same positive/negative pairs as Phase 1, over the 79-d profile; encoder Linear(79, 32) → ReLU → Dropout(0.2) → Linear(32, 16) → ReLU → Linear(16, 32); BCE on sigmoid(dot); batch 256; 90 epochs; lr 1.265711e-4 |

**Search** (`src/pipeline/tune.py`, `src/tuning/`). Optuna with a TPE sampler (seed 42), 40 trials per study, objective = mean cosine silhouette of KMeans over the K sweep {6, 12, 17, 23, 34, 46, 69}. Epoch counts are tuned in [20, 100] in steps of 10. The proposed model is tuned first; B3 and B4 are then tuned with the same objective, trial budget and epoch range, with `target_dim` fixed to the tuned `final_dim` = 32. PCA and NMF are not tuned, as they have no comparable capacity hyperparameter once the output dimension is fixed. `k_ring`, `n_neg`, batch sizes and the feature settings are held fixed.

| Study | Search space | Best mean silhouette |
|---|---|---:|
| Proposed model | `emb_dim_h` {8, 16, 32}; `emb_dim_w` {8, 16}; `emb_dim_s` {16, 32, 64}; `hidden_dim` {32, 64, 128}; `latent_dim` {8, 16}; `final_dim` {16, 32, 64}; Phase 1 and Phase 2 lr log-uniform in [1e-4, 1e-2]; epochs per phase | 0.4408 |
| B4: Hex2Vec | `hidden_dim` {32, 64, 128}; lr log-uniform in [1e-4, 1e-2]; epochs | 0.3141 |
| B3: Autoencoder | lr log-uniform in [1e-4, 1e-2]; epochs | 0.1590 |

The best-trial values above are scores of the trial's own training run (seed 42 per trial). The committed embeddings were trained separately by `train.py` with seed 32, and the baselines are retrained inside `evaluate.py` with seed 42; the resulting scores are in the next two sections and differ from the best-trial values except for B3, whose evaluate run coincides with its tuning run.

## Results

### Clustering quality

![Clustering metrics per K](src/outputs/figures/evaluation/metrics_per_k.png)

*Silhouette (higher is better), Calinski–Harabasz (higher is better) and Davies–Bouldin (lower is better) for K ∈ {6, 12, 17, 23, 34, 46, 69}. The proposed model obtains the best value of all three metrics at every K.*

Mean ± standard deviation over the K sweep, from `python -m src.pipeline.evaluate` on the committed embeddings:

| Method | Silhouette | Calinski–Harabasz | Davies–Bouldin | Dim |
|---|---:|---:|---:|---:|
| B1: PCA | +0.0310 ± 0.0536 | 40.8 ± 16.9 | 1.9952 ± 0.3923 | 32 |
| B2: NMF | +0.0826 ± 0.0171 | 84.7 ± 35.6 | 1.7022 ± 0.1239 | 32 |
| B3: Autoencoder | +0.1590 ± 0.0596 | 148.1 ± 77.4 | 1.6806 ± 0.1745 | 32 |
| B4: Hex2Vec | +0.2863 ± 0.0738 | 241.7 ± 83.3 | 1.2726 ± 0.0863 | 32 |
| M: Proposed model | +0.4052 ± 0.0649 | 684.4 ± 186.0 | 1.1149 ± 0.0994 | 32 |

### Sensitivity of NPD to k

| k | Spearman ρ vs k = 32 | Top-5% overlap |
|---:|---:|---:|
| 10 | 0.9720 | 0.8846 |
| 16 | 0.9865 | 0.9231 |
| 22 | 0.9938 | 0.9615 |
| 32 | 1.0000 | 1.0000 |
| 45 | 0.9951 | 1.0000 |

Of the 53 cells flagged at k = 32, 45 are in the top 5% for all five values of k.

### NPD and count ratio

Spearman correlation between NPD and |1 − count_ratio| over all cells: ρ = 0.358, p < 10⁻⁶.

### Highest-NPD cells

| hex_id | Alcaldía | NPD | Observed | Expected | Ratio |
|---|---|---:|---:|---:|---:|
| `8849958069fffff` | Milpa Alta | 0.5097 | 33 | 60 | 0.538 |
| `884995b229fffff` | Cuajimalpa de Morelos | 0.4730 | 45 | 65 | 0.680 |
| `884995b115fffff` | Miguel Hidalgo | 0.4678 | 36 | 80 | 0.441 |
| `884986492dfffff` | Tlalpan | 0.4557 | 32 | 52 | 0.602 |
| `8849958055fffff` | Milpa Alta | 0.4535 | 43 | 52 | 0.799 |
| `8849958607fffff` | Tlalpan | 0.3978 | 36 | 70 | 0.500 |
| `884995b945fffff` | Venustiano Carranza | 0.3976 | 221 | 315 | 0.699 |
| `88499581dbfffff` | Xochimilco | 0.3943 | 35 | 62 | 0.550 |
| `8849958f51fffff` | Tláhuac | 0.3797 | 34 | 60 | 0.556 |
| `884995ba1bfffff` | Miguel Hidalgo | 0.3706 | 226 | 551 | 0.409 |

Expected count is the similarity-weighted mean of the peers' observed counts. The lowest-NPD cells (0.017–0.022) are high-volume cells in Iztacalco, Iztapalapa, Gustavo A. Madero and Venustiano Carranza. The full ranked table is printed by `python -m src.pipeline.detect_anomalies`.

## Full clustering results

Per method and per K, from the same `evaluate` run (KMeans `random_state` 42; silhouette with cosine metric, Calinski–Harabasz and Davies–Bouldin with Euclidean distance).

| Method | K | Silhouette | Calinski–Harabasz | Davies–Bouldin |
|---|---:|---:|---:|---:|
| B1: PCA | 6 | +0.1322 | 76.3 | 2.5924 |
| B1: PCA | 12 | +0.0488 | 50.5 | 2.2893 |
| B1: PCA | 17 | +0.0592 | 42.8 | 2.2574 |
| B1: PCA | 23 | +0.0374 | 35.8 | 2.0716 |
| B1: PCA | 34 | +0.0033 | 29.7 | 1.7673 |
| B1: PCA | 46 | −0.0316 | 27.0 | 1.5485 |
| B1: PCA | 69 | −0.0322 | 23.8 | 1.4398 |
| B2: NMF | 6 | +0.0924 | 154.1 | 1.9295 |
| B2: NMF | 12 | +0.1048 | 111.9 | 1.7857 |
| B2: NMF | 17 | +0.1018 | 92.6 | 1.7031 |
| B2: NMF | 23 | +0.0766 | 74.2 | 1.7568 |
| B2: NMF | 34 | +0.0601 | 62.5 | 1.6009 |
| B2: NMF | 46 | +0.0830 | 53.7 | 1.5836 |
| B2: NMF | 69 | +0.0592 | 43.7 | 1.5556 |
| B3: Autoencoder | 6 | +0.2851 | 312.5 | 1.5459 |
| B3: Autoencoder | 12 | +0.1928 | 191.1 | 1.8211 |
| B3: Autoencoder | 17 | +0.1598 | 150.6 | 2.0021 |
| B3: Autoencoder | 23 | +0.1417 | 129.0 | 1.6558 |
| B3: Autoencoder | 34 | +0.1262 | 101.2 | 1.6583 |
| B3: Autoencoder | 46 | +0.1154 | 83.1 | 1.6661 |
| B3: Autoencoder | 69 | +0.0918 | 69.0 | 1.4151 |
| B4: Hex2Vec | 6 | +0.4245 | 401.9 | 1.1900 |
| B4: Hex2Vec | 12 | +0.3425 | 299.5 | 1.2754 |
| B4: Hex2Vec | 17 | +0.2972 | 264.7 | 1.1461 |
| B4: Hex2Vec | 23 | +0.2898 | 231.2 | 1.3098 |
| B4: Hex2Vec | 34 | +0.2407 | 189.6 | 1.3696 |
| B4: Hex2Vec | 46 | +0.2186 | 163.9 | 1.3981 |
| B4: Hex2Vec | 69 | +0.1910 | 140.8 | 1.2194 |
| M: Proposed model | 6 | +0.5464 | 1,019.5 | 0.8892 |
| M: Proposed model | 12 | +0.4322 | 841.9 | 1.0773 |
| M: Proposed model | 17 | +0.4079 | 734.0 | 1.1253 |
| M: Proposed model | 23 | +0.3880 | 671.8 | 1.1661 |
| M: Proposed model | 34 | +0.3751 | 575.6 | 1.1679 |
| M: Proposed model | 46 | +0.3427 | 506.4 | 1.1986 |
| M: Proposed model | 69 | +0.3443 | 441.3 | 1.1797 |

## Seeds and variability

- `train.py`: `torch.manual_seed(32)` and `numpy.random.seed(32)` (`ExperimentConfig.seed`); these govern negative sampling, weight initialisation, dropout and batch order.
- `evaluate.py`: `random`, `numpy` and `torch` seeded with 42 (`EvalConfig.random_state`), which also seeds PCA, NMF, KMeans and the B3/B4 training runs.
- `tune.py`: TPE sampler seeded with 42, and `torch`/`numpy` re-seeded with 42 at the start of every trial.
- KMeans uses `n_init="auto"`, which with k-means++ initialisation is a single initialisation in scikit-learn 1.8.
- Each model (proposed and baselines) is trained once. The ± values in the results tables are standard deviations across K, not across seeds; no seed-to-seed variance has been measured.
- A CPU retraining with the pinned versions reproduced the committed `final_emb.npy` to a mean per-cell cosine similarity of 0.99995 (see [Implementation details](#implementation-details)); small differences arise from floating-point non-determinism rather than from configuration.

## NPD definition

Implemented in `src/anomaly/npd.py`.

**Profile.** For each cell, the hourly distribution (24 values, normalised to sum 1), the day-of-week distribution (the first 7 columns of `weekly_matrix`, normalised to sum 1) and the 45 static features rescaled column-wise to [0, 1] with `MinMaxScaler` are concatenated into a 76-dimensional vector, which is then L2-normalised: `p_i = x_i / (‖x_i‖ + 1e-9)`. The three weekly contrast features (log weekday/weekend ratio, entropy, weekend share) are not part of the profile.

**Peers.** The k = 32 nearest cells to cell i in `final_emb` under cosine distance, excluding i itself. Similarities are `s_ij = max(0, 1 − d_ij)`, normalised to sum 1 over the k peers (`s_ij / (Σ_j s_ij + 1e-9)`).

**Score.** The expected profile is the weighted mean `p̄_i = Σ_j s_ij p_j`. It is not renormalised, so `‖p̄_i‖ ≤ 1` and

`NPD_i = 1 − p_i · p̄_i`.

Because `p̄_i` is a convex combination of unit vectors, NPD increases both when cell i's profile differs from its peers and when the peers differ from one another; it equals the cosine distance only when all peers coincide. The top 5% of cells by NPD (53 of 1,058) are flagged as candidates. The expected count is `Σ_j s_ij · count_j`, and `count_ratio = count_i / (expected_count_i + 1)`.

**Peer similarities at k = 32** (from `python -m src.pipeline.peer_stats`, all 1,058 × 32 peer pairs):

| Statistic | Cosine similarity |
|---|---:|
| Minimum | 0.967808 |
| 5th percentile | 0.996099 |
| Median | 0.999081 |
| Mean | 0.998612 |
| Maximum | 0.999968 |

The similarities are concentrated close to 1; the normalised weights `s_ij` within a peer set are therefore nearly uniform.

## Peers versus geographic neighbours

How the 32 embedding peers of each cell relate to its geographic surroundings, compared with the 32 geographically nearest cells (by centroid great-circle distance) and with 32 cells drawn at random (seed 42). Ring distances use `h3.grid_distance`; distances in km are between cell centroids.

| Neighbour set (32 per cell) | Share that are ring-1 neighbours | Share within 2 rings | Median distance (km) | Mean distance (km) |
|---|---:|---:|---:|---:|
| Embedding peers | 3.3% | 7.0% | 10.7 | 12.7 |
| Geographically nearest | 16.8% | 48.2% | 2.6 | 2.3 |
| Random cells | 0.5% | 1.4% | 16.2 | 16.8 |

Mean per-cell overlap between the embedding peers and the 32 geographically nearest cells: 11.7%.

93% of embedding peers lie more than two rings from the cell, at a median of 10.7 km. Per-cell median peer distance is mapped in `08_median_peer_distance_km.png`; three example peer sets are shown in `07_peers_examples.png`.

## Supplementary figures

These figures are produced by the pipeline and do not appear in the paper.

**NPD map.** All 1,058 cells coloured by NPD; the top 5% are outlined in black.

![NPD map](src/outputs/figures/anomaly_detection/npd_map.png)

**Case study, highest-NPD cell** (`8849958069fffff`, Milpa Alta). (A) The cell and its 32 nearest cells in embedding space. (B) Observed count against the distribution of peer counts. (C) Hourly profile of the cell against the peer mean. (D) Weekly profile against the peer mean.

![Top anomaly profile 1](src/outputs/figures/anomaly_detection/top_anomaly_profile_1.png)

The same decomposition for the second- and third-ranked cells: [`top_anomaly_profile_2.png`](src/outputs/figures/anomaly_detection/top_anomaly_profile_2.png), [`top_anomaly_profile_3.png`](src/outputs/figures/anomaly_detection/top_anomaly_profile_3.png).

**NPD against count-ratio distance.**

![NPD vs count ratio](src/outputs/figures/anomaly_detection/ratio_distance_vs_npd.png)

**Sensitivity sweep** (figure form of the table above).

![Sensitivity](src/outputs/figures/anomaly_detection/anomaly_detection_sensitivity.png)

**Training loss**, Phase 1 and Phase 2.

<img src="src/outputs/figures/phase1_loss.png" width="400"> <img src="src/outputs/figures/phase2_loss.png" width="400">

## Maps

Produced by `python -m src.pipeline.make_maps` into `src/outputs/figures/maps/` (about 20 s; also run as the last stage of `run_pipeline.py`). Basemap: CARTO Positron, which requires an API key (see [Usage](#usage)).

**Processing steps**

`01_incidents_per_cell.png` — Incidents per H3 cell on a log colour scale, with the 121 cells below the 30-incident threshold hatched in grey.

![Incidents per cell](src/outputs/figures/maps/01_incidents_per_cell.png)

`02_view_hourly_night_share.png` — Hourly view: share of each cell's incidents between 22:00 and 05:59.

![Night-time share](src/outputs/figures/maps/02_view_hourly_night_share.png)

`03_view_weekly_weekend_share.png` — Weekly view: share of each cell's incidents on Saturday and Sunday.

![Weekend share](src/outputs/figures/maps/03_view_weekly_weekend_share.png)

`04_view_static_violent_share.png` — Static view: share of each cell's incidents in a category marked as violent (`_CV`).

![Violent share](src/outputs/figures/maps/04_view_static_violent_share.png)

`05_view_static_median_delay.png` — Static view: median number of days between the incident and its report, colour clipped at the 99th percentile.

![Median reporting delay](src/outputs/figures/maps/05_view_static_median_delay.png)

**Results**

`06_kmeans_clusters_K23.png` — KMeans clusters of the final embedding at K = 23, the centre of the K sweep, with cluster sizes in the legend.

![KMeans clusters](src/outputs/figures/maps/06_kmeans_clusters_K23.png)

`07_peers_examples.png` — The 32 embedding peers of three cells: the highest-NPD cell (`8849958069fffff`, Milpa Alta, median peer distance 14.4 km), the cell closest to the centroid of all cells (`8849958581fffff`, Coyoacán, 12.8 km) and the cell farthest from it (`884995aa23fffff`, Gustavo A. Madero, 40.5 km).

![Peer examples](src/outputs/figures/maps/07_peers_examples.png)

`08_median_peer_distance_km.png` — Median centroid distance from each cell to its 32 embedding peers.

![Median peer distance](src/outputs/figures/maps/08_median_peer_distance_km.png)

`09_npd_candidates.png` — NPD at k = 32 with the 53 top-5% candidates outlined.

![NPD candidates](src/outputs/figures/maps/09_npd_candidates.png)

`10_candidate_stability.png` — For each cell, the number of values of k ∈ {10, 16, 22, 32, 45} at which it is in the top 5%; 45 cells are flagged at all five.

![Candidate stability](src/outputs/figures/maps/10_candidate_stability.png)

`11_count_ratio.png` — Observed over expected incident count per cell on a log colour scale centred at 1 (red above, blue below).

![Count ratio](src/outputs/figures/maps/11_count_ratio.png)

## Repository layout

```
run_pipeline.py              entry point: train → evaluate → detect_anomalies → make_maps
src/
  config.py                  paths and hyperparameters (dataclasses)
  data/
    loader.py                CSV loading, dtypes, temporal features
    taxonomy.py              CRIME_TAXONOMY: delito text → 38 categories
  features/
    hex_features.py          H3 assignment and per-cell feature matrices
  models/
    phase1.py                per-view encoders, geographic pair sampling
    phase2.py                autoencoders and attention fusion
    baselines.py             PCA, NMF, dense autoencoder, Hex2Vec
  pipeline/
    train.py                 Phase 1 + Phase 2, writes embeddings
    evaluate.py              clustering benchmark
    detect_anomalies.py      NPD, sensitivity, figures
    make_maps.py             maps of processing steps and results
    peer_stats.py            peer similarity and peers-vs-geography statistics
    tune.py                  Optuna search
    run_all.py               stage orchestration
  evaluation/                clustering metrics and report tables
  anomaly/
    npd.py                   NPD computation
    peers.py                 embedding peers and geographic comparison
    plots.py                 anomaly figures and basemap helper
  tuning/                    Optuna search space and objective
  common/                    logging, artifact I/O, plotting helpers
  outputs/
    embeddings/              final_emb.npy, E_h/E_w/E_s.npy, hex_ids.json, metadata.json
    figures/                 all figures shown above
    tuning/best_params.json  Optuna results
data/
  carpetasFGJ_pipeline.csv   FGJ dataset (Git LFS)
```

Everything under `src/outputs/` is regenerated by a full pipeline run.

## Installation

Python 3.13. Main dependencies: PyTorch 2.12, h3 4.4, scikit-learn 1.8, GeoPandas 1.1, Optuna 5.0, contextily 1.7. CPU is sufficient; CUDA is used when available.

```bash
git clone <repository-url> urban-npd-cdmx
cd urban-npd-cdmx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (about 560 MB) is stored with Git LFS:

```bash
git lfs install
git lfs pull
```

Map figures use the CARTO Positron basemap, which requires an API key. Put it in a `.env` file at the repository root (ignored by git) or export it:

```
CARTO_API_KEY=<your key>
```

Without a key the maps are drawn without a basemap and a warning is logged.

## Usage

```bash
python run_pipeline.py                                 # train → evaluate → detect_anomalies → make_maps
python run_pipeline.py --skip-train --skip-evaluate    # detection and maps only, using saved embeddings

python -m src.pipeline.train
python -m src.pipeline.evaluate
python -m src.pipeline.detect_anomalies
python -m src.pipeline.make_maps
python -m src.pipeline.peer_stats                      # peer similarity and geography statistics
python -m src.pipeline.tune                            # hyperparameter search (slow)
```

Trained embeddings for the current run are committed under `src/outputs/embeddings/`. `hex_ids.json` gives the row order of `final_emb.npy`, so the embedding can be joined to H3 cell IDs without retraining.

## Configuration

All settings are in `src/config.py`.

| Parameter | Value | Description |
|---|---|---|
| `features.resolution` | 8 | H3 resolution |
| `features.min_events` | 30 | minimum events per cell |
| `phase1.k_ring` | 1 | ring distance for positive pairs |
| `phase1.n_neg` | 3 | negatives per cell |
| `phase1.emb_dim_h/w/s` | 32 / 8 / 64 | Phase 1 embedding sizes |
| `phase1.hidden_dim` | 32 | Phase 1 encoder hidden width (the search's best trial used 64; see [Implementation details](#implementation-details)) |
| `phase2.latent_dim` | 8 | autoencoder latent size |
| `phase2.final_dim` | 32 | final embedding size |
| `anomaly.k_value` | 32 | peers per cell for NPD |
| `anomaly.top_percentile` | 0.05 | fraction flagged as candidates |
| `seed` | 32 | training seed (42 for evaluation and tuning) |

Learning rates and epoch counts in `config.py` are the values found by the Optuna search in `src/outputs/tuning/best_params.json` (40 trials). `eval.k_values` is a fallback; the K sweep is computed from N by `suggest_k_values()` at run time.

## Data

`data/carpetasFGJ_pipeline.csv` contains investigation-file records (*carpetas de investigación*) published by FGJ-CDMX on the Mexico City open data portal:

- Source: https://datos.cdmx.gob.mx/dataset/carpetas-de-investigacion-fgj-de-la-ciudad-de-mexico
- Download date: TODO
- Records: 2,098,743 (the file has 2,102,548 lines including the header; 3,804 records contain line breaks inside quoted fields)
- Report dates (`fecha_inicio`): 2016-01-01 to 2025-01-31; incident dates (`fecha_hecho`) extend earlier for incidents reported long after they occurred
- SHA-256: `6335d8802459f0d7fe1cd66e0402674ada6ead0f5f609952b40c82e90f64134c`

Columns: year/month/date/time of the report (`*_inicio`) and of the incident (`*_hecho`), offence description (`delito`), offence category, jurisdiction, prosecutor's office, agency and investigation unit, neighbourhood and borough of the incident, municipality, latitude and longitude. No personal identifiers are included.

The file is used as downloaded; all filtering and mapping happens in `src/data/loader.py`. Offence descriptions not present in `CRIME_TAXONOMY` are dropped at load time; the loader logs the number of dropped rows.

The exact file is versioned in this repository with Git LFS (`.gitattributes`; the LFS object ID equals the SHA-256 above) because the portal dataset is updated monthly and earlier versions are not kept. GitHub applies bandwidth and storage quotas to LFS downloads; a copy of the file archived on Zenodo, with a DOI, would provide a permanent reference independent of those quotas.

## Interpretation

The data record reports, not incidents. A cell with few reports may have few incidents, limited access to reporting, or low institutional trust; the data do not distinguish these.

NPD identifies cells whose reporting profile differs from that of cells the model considers similar. Land use, daytime versus resident population, commercial density and reporting infrastructure are not modelled and can produce high NPD in the absence of any reporting disparity. A high score indicates a cell to examine with additional data, not a finding about that cell.

## Limitations

- The embedding and the NPD profile are built from the same three feature matrices, with different normalisation. Cells are therefore close in embedding space in part because their profiles are similar, which reduces the residual that NPD measures. The Phase 1 objective is geographic rather than reconstructive, so the embedding does not reproduce the profile directly, but the overlap should be taken into account when reading NPD magnitudes.
- Median reporting delay and same-day report fraction are included in `static_matrix` and therefore in the encoder input. The results reported here were produced with these features present. Removing them from the encoder input while keeping them in the compared profile is planned.
- The expected profile is a weighted mean of unit-norm vectors and is not renormalised, so NPD increases with disagreement among peers as well as with the cell's own divergence (see [NPD definition](#npd-definition)).
- Peer cosine similarities in the final embedding are concentrated near 1 (median 0.9991), so the similarity weights inside a peer set are nearly uniform and the score depends on which cells are selected rather than on how strongly they are weighted.
- The committed embeddings use `phase1.hidden_dim = 32` rather than the 64 selected by the hyperparameter search.
- Each model is trained once; no variance across seeds is reported.
- There are no automated tests. Changes to features or models should be checked by rerunning the pipeline and inspecting `src/outputs/`.
- No license file is included yet.

## Preprint

Available in arXiv [HERE](https://arxiv.org/abs/2611.21825).

## Citation

```bibtex
@inproceedings{valverde2027urban,
  title = {Urban Representation Learning for Identifying Potential Disparities in Crime Reporting},
  author = {Valverde-Rebaza, Jorge and Paez De La Cruz, C. and P{\'e}rez-Collado, P. and S{\'a}nchez-Cata{\~n}o, D. and Mart{\'\i}nez-Vergara, E.},
  booktitle = {Proceedings of the International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2027)},
  year = {2027},
  publisher = {IEEE},
  note = {Accepted, to be published},
}
```

## Acknowledgments

Data: Fiscalía General de Justicia de la Ciudad de México, via the Mexico City open data portal. Spatial indexing: [H3](https://h3geo.org/). Basemaps in figures: OpenStreetMap contributors, CARTO.
