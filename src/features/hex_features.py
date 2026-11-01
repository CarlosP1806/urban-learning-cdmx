import numpy as np
import pandas as pd
import h3
from scipy.stats import entropy as shannon_entropy
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import torch

from src.config import FeatureConfig

RESOLUTION = FeatureConfig().resolution
MIN_EVENTS = FeatureConfig().min_events

@dataclass
class HexFeatures:
    hourly_matrix:   np.ndarray          # (N, 24)
    weekly_matrix:   np.ndarray          # (N, 10)
    static_matrix:   np.ndarray          # (N, C+static features)
    hex_ids:         list                # ordered list of hex_id
    hex_to_idx:      dict                # hex_id -> index in the matrices
    crime_types:     list                # unique crime categories
    hex_crime_count: dict                # hex_id -> total events
    hex_to_alcaldia: dict                # hex_id -> most frequent alcaldia
    N:               int                 # number of hexagons
    C:               int                 # number of crime categories


def assign_hex_ids(df: pd.DataFrame, resolution: int = RESOLUTION) -> pd.Series:
    return pd.Series(
        [h3.latlng_to_cell(lat, lon, resolution)
         for lat, lon in zip(df["latitud"], df["longitud"])],
        index=df.index,
    )


def build_hex_features(df: pd.DataFrame,
                       resolution: int = RESOLUTION,
                       min_events: int = MIN_EVENTS) -> HexFeatures:
    df = df.copy()
    if "hex_id" not in df.columns:
        df["hex_id"] = assign_hex_ids(df, resolution)

    counts      = df["hex_id"].value_counts()
    valid       = counts[counts >= min_events].index
    df          = df[df["hex_id"].isin(valid)]

    hex_ids     = sorted(df["hex_id"].unique().tolist())
    hex_to_idx  = {h: i for i, h in enumerate(hex_ids)}
    N           = len(hex_ids)

    crime_types  = sorted(df["crime_type"].unique().tolist())
    C            = len(crime_types)

    # Initialize matrices
    hourly_matrix = np.zeros((N, 24))
    weekly_matrix = np.zeros((N, 10))       # 7 days + 3 contrast features
    static_matrix = np.zeros((N, C + 7))

    for hex_id, group in df.groupby("hex_id"):
        idx   = hex_to_idx[hex_id]
        total = len(group)

        # Hourly
        for h, cnt in group["hour"].value_counts().items():
            hourly_matrix[idx, h] = cnt / total

        # Weekly
        dow_dist = (group["dow"]
                    .value_counts(normalize=True)
                    .reindex(range(7), fill_value=0))
        weekly_matrix[idx, :7] = dow_dist.values
        weekday = dow_dist[[0,1,2,3,4]].sum()
        weekend = dow_dist[[5,6]].sum()
        weekly_matrix[idx, 7] = np.log((weekday + 1e-9) / (weekend + 1e-9))
        weekly_matrix[idx, 8] = shannon_entropy(dow_dist.values + 1e-9)
        weekly_matrix[idx, 9] = weekend

        # Static
        ptr      = 0
        type_dist = (group["crime_type"]
                     .value_counts(normalize=True)
                     .reindex(crime_types, fill_value=0))
        static_matrix[idx, ptr:ptr+C] = type_dist.values
        ptr += C

        static_matrix[idx, ptr] = group["crime_type"].str.contains(
            "CV", na=False).mean();                ptr += 1
        static_matrix[idx, ptr] = shannon_entropy(
            type_dist.values + 1e-9);                         ptr += 1
        hour_dist = (group["hour"]
                     .value_counts(normalize=True)
                     .reindex(range(24), fill_value=0).values)
        static_matrix[idx, ptr] = shannon_entropy(
            hour_dist + 1e-9);                                ptr += 1
        static_matrix[idx, ptr] = group["crime_type"].isin([
            "HOMICIDIO_DOLOSO", "DELITOS_SEXUALES", "SECUESTRO",
            "FEMINICIDIO", "LESIONES_INTENCIONALES"
        ]).mean();                                            ptr += 1

        if "fecha_inicio" in group.columns:
            days = (group["fecha_inicio"] - group["fecha_hecho"]).dt.days.clip(lower=0)
            static_matrix[idx, ptr]   = days.median()
            static_matrix[idx, ptr+1] = (days == 0).mean()
        ptr += 2

        yearly_count = group.groupby("anio_hecho").size()
        if len(yearly_count) >= 3:
            years      = yearly_count.index.values.astype(float)
            counts_y   = yearly_count.values.astype(float)
            years_norm = (years - years.mean()) / (years.std() + 1e-9)
            slope      = np.polyfit(years_norm, counts_y, 1)[0]
            static_matrix[idx, ptr] = slope / (counts_y.mean() + 1e-9)
        ptr += 1

    hex_crime_count = df.groupby("hex_id").size().to_dict()
    hex_to_alcaldia = (
        df.groupby("hex_id")["alcaldia_hecho"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    return HexFeatures(
        hourly_matrix   = hourly_matrix,
        weekly_matrix   = weekly_matrix,
        static_matrix   = static_matrix,
        hex_ids         = hex_ids,
        hex_to_idx      = hex_to_idx,
        crime_types     = crime_types,
        hex_crime_count = hex_crime_count,
        hex_to_alcaldia = hex_to_alcaldia,
        N               = N,
        C               = C,
    )

def to_tensors(hf: HexFeatures, device):
    def scale(m):
        return torch.tensor(
            StandardScaler().fit_transform(m),
            dtype=torch.float32
        ).to(device)

    return scale(hf.hourly_matrix), scale(hf.weekly_matrix), scale(hf.static_matrix)
