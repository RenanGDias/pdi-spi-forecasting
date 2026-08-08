"""Métricas numéricas e categóricas usadas no artigo e na auditoria."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score


SPI_BINS = np.array([-np.inf, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, np.inf])


def spi_classes(values: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), SPI_BINS[1:-1], right=True)


def pearson_r(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[mask], predicted[mask]
    if observed.size < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[mask], predicted[mask]
    if observed.size == 0:
        return {key: float("nan") for key in ("r", "r2", "kappa", "mae", "rmse")}
    return {
        "r": pearson_r(observed, predicted),
        "r2": float(r2_score(observed, predicted)) if observed.size > 1 else float("nan"),
        "kappa": float(cohen_kappa_score(spi_classes(observed), spi_classes(predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
    }

