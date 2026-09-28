"""Small, leak-aware feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

_DATE_PARTS = ("year", "month", "day", "dayofweek", "dayofyear", "hour", "minute")


def date_features(
    df: pd.DataFrame, col: str, tz: str | None = None, cyclical: bool = True
) -> pd.DataFrame:
    """Add calendar features for datetime column ``col`` (optionally converted to ``tz``).

    When ``tz`` is given, naive timestamps are assumed to be UTC before conversion;
    without ``tz`` they are used as-is. Missing timestamps (NaT) give missing features.
    Cyclical encodings (sin/cos) are added for month, day of week and hour, so that
    December sits next to January.
    """
    ts = pd.to_datetime(df[col])
    if tz is not None:
        ts = (ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts).dt.tz_convert(tz)
    out = df.copy()
    for part in _DATE_PARTS:
        out[f"{col}_{part}"] = getattr(ts.dt, part)
    out[f"{col}_week"] = ts.dt.isocalendar().week.astype("Int32")  # nullable: NaT-safe
    out[f"{col}_is_weekend"] = (ts.dt.dayofweek >= 5).astype("Int8").mask(ts.isna())
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

    Rows with fold ``-1`` (train-only) are never validated, so they are encoded
    leave-one-out: from every labelled row except themselves.
    """
    prior = df[target].mean()
    encoded = np.full(len(df), prior, dtype=float)

    def fit(train: pd.DataFrame) -> pd.Series:
        stats = train.groupby(col)[target].agg(["sum", "count"])
        return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)

    folds = np.asarray(folds)
    for fold in np.unique(folds):
        valid = folds == fold
        if fold == -1:
            stats = df.groupby(col)[target].agg(["sum", "count"])
            rows = df.loc[valid]
            total = rows[col].map(stats["sum"]) - rows[target]
            count = rows[col].map(stats["count"]) - 1
            denom = count + smoothing
            loo = (total + prior * smoothing) / denom.where(denom > 0)
            encoded[valid] = loo.fillna(prior).to_numpy()
        else:
            mapping = fit(df[~valid])
            encoded[valid] = df.loc[valid, col].map(mapping).fillna(prior).to_numpy()
    return encoded
