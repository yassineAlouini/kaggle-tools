"""Small, leak-aware feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

_DATE_PARTS = ("year", "month", "day", "dayofweek", "dayofyear", "hour", "minute")


def date_features(
    df: pd.DataFrame, col: str, tz: str | None = None, cyclical: bool = True
) -> pd.DataFrame:
    """Add calendar features for datetime column ``col`` (optionally converted to ``tz``).

    Naive timestamps are assumed to be UTC. Cyclical encodings (sin/cos) are added
    for month, day of week and hour, so that December sits next to January.
    """
    ts = pd.to_datetime(df[col])
    if tz is not None:
        ts = (ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts).dt.tz_convert(tz)
    out = df.copy()
    for part in _DATE_PARTS:
        out[f"{col}_{part}"] = getattr(ts.dt, part)
    out[f"{col}_week"] = ts.dt.isocalendar().week.astype("int32")
    out[f"{col}_is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    if cyclical:
        for part, period in (("month", 12), ("dayofweek", 7), ("hour", 24)):
            angle = 2 * np.pi * getattr(ts.dt, part) / period
            out[f"{col}_{part}_sin"] = np.sin(angle)
            out[f"{col}_{part}_cos"] = np.cos(angle)
    return out


def oof_target_encode(
    df: pd.DataFrame, col: str, target: str, folds: np.ndarray, smoothing: float = 10.0
) -> np.ndarray:
    """Out-of-fold, smoothed mean target encoding (no target leakage into train rows).

    Rows with fold ``-1`` (train-only) are encoded using all labelled rows.
    """
    prior = df[target].mean()
    encoded = np.full(len(df), prior, dtype=float)

    def fit(train: pd.DataFrame) -> pd.Series:
        stats = train.groupby(col)[target].agg(["sum", "count"])
        return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)

    for fold in np.unique(folds):
        valid = folds == fold
        train = df[~valid] if fold != -1 else df
        mapping = fit(train)
        encoded[valid] = df.loc[valid, col].map(mapping).fillna(prior).to_numpy()
    return encoded
