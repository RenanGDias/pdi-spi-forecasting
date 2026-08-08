"""Cálculo causal ou global do Standardized Precipitation Index (SPI)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gamma, norm


def _fit_gamma_spi(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    calibration = calibration[np.isfinite(calibration)]
    if calibration.size < 4:
        return np.full(values.shape, np.nan, dtype=float)
    positive = calibration[calibration > 0]
    if positive.size < 4 or np.allclose(positive, positive[0]):
        return np.full(values.shape, np.nan, dtype=float)

    zero_probability = float(np.mean(calibration <= 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape, _, scale = gamma.fit(positive, floc=0)

    safe = np.maximum(values, 0.0)
    cumulative = zero_probability + (1.0 - zero_probability) * gamma.cdf(
        safe, a=shape, loc=0, scale=scale
    )
    cumulative = np.clip(cumulative, 1e-7, 1.0 - 1e-7)
    result = norm.ppf(cumulative)
    result[~np.isfinite(values)] = np.nan
    return result


def compute_spi(
    precipitation: xr.DataArray,
    scale: int,
    calibration_start: str = "1998-01-01",
    calibration_end: str = "2015-12-31",
    min_periods: int | None = None,
) -> xr.DataArray:
    """Calcula SPI-gama por mês do ano e ponto espacial.

    O intervalo de calibração é explícito. Usar ``calibration_end='2013-12-31'``
    impede que validação (2014) e teste (2015) influenciem a transformação.
    """
    if scale < 1:
        raise ValueError("A escala do SPI deve ser positiva.")
    required = {"time", "lat", "lon"}
    if set(precipitation.dims) != required:
        raise ValueError(f"Dimensões esperadas {required}; recebidas {set(precipitation.dims)}")

    effective_min_periods = scale if min_periods is None else min_periods
    if not 1 <= effective_min_periods <= scale:
        raise ValueError("min_periods deve estar entre 1 e scale")
    accumulated = precipitation.rolling(time=scale, min_periods=effective_min_periods).sum()
    times = pd.DatetimeIndex(accumulated.time.values)
    calibration_mask = (times >= pd.Timestamp(calibration_start)) & (
        times <= pd.Timestamp(calibration_end)
    )
    flat = accumulated.values.reshape(accumulated.sizes["time"], -1)
    transformed = np.full_like(flat, np.nan, dtype=float)

    for month in range(1, 13):
        month_mask = times.month == month
        fit_mask = month_mask & calibration_mask
        for point in range(flat.shape[1]):
            transformed[month_mask, point] = _fit_gamma_spi(
                flat[fit_mask, point], flat[month_mask, point]
            )

    result = xr.DataArray(
        transformed.reshape(accumulated.shape),
        coords=accumulated.coords,
        dims=accumulated.dims,
        name=f"spi_{scale}",
        attrs={
            "scale_months": scale,
            "distribution": "gamma_mle_zero_adjusted",
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            "min_periods": effective_min_periods,
        },
    )
    return result
