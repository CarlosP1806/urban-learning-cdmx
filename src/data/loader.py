import pandas as pd
from pathlib import Path

from src.data.taxonomy import CRIME_TAXONOMY
from src.common.log import get_logger, fmt_int

logger = get_logger(__name__)

DATE_COLS  = ["fecha_inicio", "fecha_hecho"]
TIME_COLS  = ["hora_inicio", "hora_hecho"]
MONTH_COLS = ["mes_inicio", "mes_hecho"]
INT_COLS   = ["anio_inicio", "anio_hecho"]
FLOAT_COLS = ["latitud", "longitud"]
TEXT_COLS  = [
    "delito", "categoria_delito", "competencia", "fiscalia",
    "agencia", "unidad_investigacion", "colonia_hecho", "colonia_catalogo",
    "alcaldia_hecho", "alcaldia_catalogo", "municipio_hecho"
]


def load(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load the dataset from a CSV file, applying dtypes and extracting temporal features.
    """
    df = pd.read_csv(path, low_memory=False)
    if verbose:
        logger.info(f"Loaded {fmt_int(len(df))} raw rows from {Path(path).name}")

    df = _apply_dtypes(df, verbose)
    df = _add_temporal_features(df, verbose)

    if verbose:
        logger.info(f"{fmt_int(len(df))} rows remain after cleaning and taxonomy mapping")
    return df


def _apply_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Apply specific dtypes to the DataFrame's columns, handling errors and reporting missing columns.
    """
    expected = set(DATE_COLS + TIME_COLS + MONTH_COLS +
                   INT_COLS + FLOAT_COLS + TEXT_COLS)
    if verbose:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")

    for c in [x for x in TEXT_COLS + MONTH_COLS if x in df.columns]:
        df[c] = df[c].astype("string")

    for c in [x for x in INT_COLS if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in [x for x in FLOAT_COLS if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")

    for c in [x for x in DATE_COLS if x in df.columns]:
        df[c] = (df[c].str.extract(r"(\d{4}-\d{2}-\d{2})")
                      .apply(pd.to_datetime, errors="coerce", dayfirst=False))

    for c in [x for x in TIME_COLS if x in df.columns]:
        df[c] = (df[c].str.extract(r"(\d{2}:\d{2})")
                      .apply(pd.to_datetime, errors="coerce", dayfirst=False))
    return df


def _add_temporal_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add temporal features to the DataFrame, such as hour of day and day of week,
    and drop rows with missing data.
    """
    n_before = len(df)
    df = df.dropna(subset=["latitud", "longitud", "fecha_hecho", "hora_hecho"])
    df = df.copy()
    df["hour"] = df["hora_hecho"].dt.hour
    df["dow"]  = df["fecha_hecho"].dt.dayofweek
    if verbose and len(df) < n_before:
        logger.info(f"Dropped {fmt_int(n_before - len(df))} rows with missing location/date")

    n_before = len(df)
    df["crime_type_raw"] = df["delito"].fillna("").str.upper().str.strip()
    df["crime_type"]     = df["crime_type_raw"].map(CRIME_TAXONOMY)
    df                   = df[df["crime_type"].notna()]
    if verbose and len(df) < n_before:
        logger.info(f"Dropped {fmt_int(n_before - len(df))} rows with unmapped/irrelevant crime types")

    return df
