"""IO helpers for saving/loading DataFrames."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PathLike = str | os.PathLike


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, path: PathLike, fmt: str = "parquet") -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if fmt == "parquet":
        df.to_parquet(p)
    elif fmt == "csv":
        df.to_csv(p)
    else:
        raise ValueError(f"Unknown format {fmt}")
    return p


def load_dataframe(path: PathLike) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p, index_col=0, parse_dates=True)
    raise ValueError(f"Unknown extension: {p.suffix}")
