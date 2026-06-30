""" Data pipeline with Polars: load, clean, perpare for DoubleML """

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class PreparedData:
    X: np.ndarray
    D: np.ndarray
    Y: np.ndarray
    feature_names: list[str]
    n_obs: int
    n_features: int


def load_and_validate(path: str | Path, treatment_col: str, outcome_col: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    missing = {treatment_col, outcome_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Fill numeric nulls with median, one-hot encode low-cardinality strings,
    drop high-cardinality (free-text) string columns."""

    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
    ]
    for col in numeric_cols:
        df = df.with_columns(pl.col(col).fill_null(df[col].median()))

    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    # Drop free-text columns (too many unique values to one-hot encode)
    high_card = [c for c in string_cols if df[c].n_unique() > 20]
    if high_card:
        df = df.drop(high_card)
        string_cols = [c for c in string_cols if c not in high_card]

    if string_cols:
        df = df.to_dummies(columns=string_cols, drop_first=True)

    return df


def prepare_for_doubleml(df: pl.DataFrame, treatment_col: str, outcome_col: str) -> PreparedData:
    feature_cols = [c for c in df.columns if c not in (treatment_col, outcome_col)]
    X = df.select(feature_cols).to_numpy()
    D = df[treatment_col].to_numpy().flatten()
    Y = df[outcome_col].to_numpy().flatten()
    return PreparedData(
        X=X, D=D, Y=Y,
        feature_names=feature_cols,
        n_obs=X.shape[0],
        n_features=X.shape[1],
    )


def run_pipeline(path: str | Path, treatment_col: str = "treatment", outcome_col: str = "outcome") -> PreparedData:
    df = load_and_validate(path, treatment_col, outcome_col)
    df = clean_data(df)
    return prepare_for_doubleml(df, treatment_col, outcome_col)

