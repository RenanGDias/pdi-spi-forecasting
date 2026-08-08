"""Avaliação estratificada por mês, escala e subconjunto."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def metrics_by_month(
    observed: np.ndarray,
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    extra: dict | None = None,
) -> pd.DataFrame:
    records = []
    frame = metadata.reset_index(drop=True)
    for target_time, group in frame.groupby("target_time", sort=True):
        indices = group.index.to_numpy()
        record = {"target_time": pd.Timestamp(target_time), "month": pd.Timestamp(target_time).month}
        record.update(regression_metrics(observed[indices], predicted[indices]))
        if extra:
            record.update(extra)
        records.append(record)
    return pd.DataFrame(records)


def metrics_by_split(
    observed: np.ndarray,
    predicted: np.ndarray,
    split_indices: dict[str, np.ndarray],
    extra: dict | None = None,
) -> pd.DataFrame:
    records = []
    for split, indices in split_indices.items():
        record = {"split": split, "n": len(indices)}
        record.update(regression_metrics(observed[indices], predicted[indices]))
        if extra:
            record.update(extra)
        records.append(record)
    all_record = {"split": "all", "n": len(observed)}
    all_record.update(regression_metrics(observed, predicted))
    if extra:
        all_record.update(extra)
    records.append(all_record)
    return pd.DataFrame(records)

