"""Reconstruções concorrentes do desenho do artigo e desenhos causais."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def _field_at(data: xr.DataArray, year: int, month: int) -> np.ndarray:
    selected = data.sel(time=f"{year:04d}-{month:02d}-01")
    return np.asarray(selected.values, dtype=float)


def make_paper_like_design(
    spi: xr.DataArray,
    input_year_count: int = 15,
    target_year: int = 2015,
) -> DesignMatrix:
    """Interpretação da Figura 4: anos são atributos; mês-ponto é amostra.

    Para ANN#15, cada amostra tem 15 atributos (o mesmo mês/ponto entre
    2000 e 2014), e o rótulo é o valor correspondente em 2015. Isso permite
    testar se a divisão 70/15/15 usa rótulos do próprio ano anunciado como
    previsão.
    """
    years = list(range(target_year - input_year_count, target_year))
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    metadata: list[dict] = []
    latitudes = np.asarray(spi.lat.values)
    longitudes = np.asarray(spi.lon.values)

    for month in range(1, 13):
        features = np.stack([_field_at(spi, year, month).ravel() for year in years], axis=1)
        target = _field_at(spi, target_year, month).ravel()
        rows_x.append(features)
        rows_y.append(target)
        for point_index, (lat_index, lon_index) in enumerate(
            np.ndindex(spi.sizes["lat"], spi.sizes["lon"]), start=1
        ):
            metadata.append(
                {
                    "target_time": pd.Timestamp(target_year, month, 1),
                    "month": month,
                    "point_id": point_index,
                    "lat": float(latitudes[lat_index]),
                    "lon": float(longitudes[lon_index]),
                }
            )

    X = np.concatenate(rows_x, axis=0)
    y = np.concatenate(rows_y, axis=0)
    meta = pd.DataFrame(metadata)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return DesignMatrix(X=X[finite], y=y[finite], metadata=meta.loc[finite].reset_index(drop=True))


def sequential_indices(
    size: int, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> dict[str, np.ndarray]:
    train_end = int(np.floor(size * train_fraction))
    validation_end = train_end + int(np.floor(size * validation_fraction))
    indices = np.arange(size)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def make_causal_tabular_design(
    spi: xr.DataArray,
    lookback_months: int = 12,
    horizon_months: int = 1,
) -> DesignMatrix:
    """Cria amostras usando somente campos anteriores à data prevista."""
    if lookback_months < 1 or horizon_months < 1:
        raise ValueError("lookback_months e horizon_months devem ser positivos")

    values = np.asarray(spi.values, dtype=float)
    times = pd.DatetimeIndex(spi.time.values)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    metadata: list[dict] = []
    lats = np.asarray(spi.lat.values, dtype=float)
    lons = np.asarray(spi.lon.values, dtype=float)

    first_target = lookback_months + horizon_months - 1
    for target_index in range(first_target, len(times)):
        input_end = target_index - horizon_months
        input_start = input_end - lookback_months + 1
        history = values[input_start : input_end + 1]
        target = values[target_index]
        month_angle = 2.0 * np.pi * (times[target_index].month - 1) / 12.0

        for point_index, (lat_index, lon_index) in enumerate(
            np.ndindex(spi.sizes["lat"], spi.sizes["lon"]), start=1
        ):
            temporal = history[:, lat_index, lon_index]
            feature = np.concatenate(
                [
                    temporal,
                    [np.sin(month_angle), np.cos(month_angle)],
                    [lats[lat_index], lons[lon_index]],
                ]
            )
            rows_x.append(feature)
            rows_y.append(np.asarray(target[lat_index, lon_index]))
            metadata.append(
                {
                    "target_time": times[target_index],
                    "input_end": times[input_end],
                    "horizon_months": horizon_months,
                    "point_id": point_index,
                    "lat": lats[lat_index],
                    "lon": lons[lon_index],
                }
            )

    X = np.asarray(rows_x, dtype=float)
    y = np.asarray(rows_y, dtype=float).reshape(-1)
    meta = pd.DataFrame(metadata)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return DesignMatrix(X=X[finite], y=y[finite], metadata=meta.loc[finite].reset_index(drop=True))


def make_causal_annual_design(
    spi: xr.DataArray,
    lookback_months: int = 12,
    forecast_months: int = 12,
) -> DesignMatrix:
    """Prevê, de uma única origem, os doze meses seguintes por ponto da grade."""
    values = np.asarray(spi.values, dtype=float)
    times = pd.DatetimeIndex(spi.time.values)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    metadata: list[dict] = []
    lats = np.asarray(spi.lat.values, dtype=float)
    lons = np.asarray(spi.lon.values, dtype=float)

    first_origin = lookback_months - 1
    last_origin = len(times) - forecast_months - 1
    for origin_index in range(first_origin, last_origin + 1):
        input_start = origin_index - lookback_months + 1
        history = values[input_start : origin_index + 1]
        target = values[origin_index + 1 : origin_index + 1 + forecast_months]
        origin = times[origin_index]
        for point_index, (lat_index, lon_index) in enumerate(
            np.ndindex(spi.sizes["lat"], spi.sizes["lon"]), start=1
        ):
            rows_x.append(history[:, lat_index, lon_index])
            rows_y.append(target[:, lat_index, lon_index])
            metadata.append(
                {
                    "origin_time": origin,
                    "target_start": times[origin_index + 1],
                    "target_end": times[origin_index + forecast_months],
                    "point_id": point_index,
                    "lat": lats[lat_index],
                    "lon": lons[lon_index],
                }
            )

    X = np.asarray(rows_x, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    meta = pd.DataFrame(metadata)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    return DesignMatrix(X=X[finite], y=y[finite], metadata=meta.loc[finite].reset_index(drop=True))


def annual_origin_indices(
    metadata: pd.DataFrame,
    train_target_end: str = "2013-12-31",
    validation_origin: str = "2013-12-01",
    test_origin: str = "2014-12-01",
) -> dict[str, np.ndarray]:
    target_end = pd.DatetimeIndex(metadata["target_end"])
    origin = pd.DatetimeIndex(metadata["origin_time"])
    return {
        "train": np.flatnonzero(target_end <= pd.Timestamp(train_target_end)),
        "validation": np.flatnonzero(origin == pd.Timestamp(validation_origin)),
        "test": np.flatnonzero(origin == pd.Timestamp(test_origin)),
    }


def temporal_indices(
    metadata: pd.DataFrame,
    train_end: str = "2013-12-31",
    validation_start: str = "2014-01-01",
    validation_end: str = "2014-12-31",
    test_start: str = "2015-01-01",
    test_end: str = "2015-12-31",
) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(metadata["target_time"])
    return {
        "train": np.flatnonzero(dates <= pd.Timestamp(train_end)),
        "validation": np.flatnonzero(
            (dates >= pd.Timestamp(validation_start)) & (dates <= pd.Timestamp(validation_end))
        ),
        "test": np.flatnonzero(
            (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
        ),
    }


def make_spatial_sequences(
    spi: xr.DataArray,
    lookback_months: int = 12,
    horizon_months: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Converte mapas de SPI em pares imagem-sequência para a CNN espacial."""
    values = np.asarray(spi.values, dtype=np.float32)
    times = pd.DatetimeIndex(spi.time.values)
    inputs, targets, target_times = [], [], []
    first_target = lookback_months + horizon_months - 1
    for target_index in range(first_target, len(times)):
        input_end = target_index - horizon_months
        input_start = input_end - lookback_months + 1
        x = values[input_start : input_end + 1]
        y = values[target_index : target_index + 1]
        if np.isfinite(x).all() and np.isfinite(y).all():
            inputs.append(x)
            targets.append(y)
            target_times.append(times[target_index])
    return np.asarray(inputs), np.asarray(targets), pd.DatetimeIndex(target_times)
