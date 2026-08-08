"""Leitura e validação dos dados mensais TRMM extraídos para a grade do artigo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_COLUMNS = {"date", "lat", "lon", "precipitation_mm"}


def load_monthly_trmm_csv(path: str | Path) -> xr.DataArray:
    """Carrega o CSV mensal e devolve precipitação com dimensões tempo/lat/lon."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    frame = frame.loc[:, ["date", "lat", "lon", "precipitation_mm"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    frame[["lat", "lon", "precipitation_mm"]] = frame[
        ["lat", "lon", "precipitation_mm"]
    ].apply(pd.to_numeric, errors="raise")

    duplicate = frame.duplicated(["date", "lat", "lon"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["date", "lat", "lon"]].head().to_dict("records")
        raise ValueError(f"Há observações duplicadas: {examples}")

    series = frame.set_index(["date", "lat", "lon"])["precipitation_mm"].sort_index()
    data = series.to_xarray().rename(date="time").transpose("time", "lat", "lon")
    data.name = "precipitation_mm"
    data.attrs.update(
        units="mm/month",
        source="TRMM 3B42 V7, monthly sum of 3-hourly rates multiplied by 3 h",
    )
    validate_article_grid(data)
    return data


def validate_article_grid(data: xr.DataArray) -> None:
    """Falha cedo quando o arquivo não corresponde à grade/período declarados."""
    expected_dims = {"time", "lat", "lon"}
    if set(data.dims) != expected_dims:
        raise ValueError(f"Dimensões esperadas {expected_dims}; recebidas {set(data.dims)}")
    if data.sizes["lat"] != 13 or data.sizes["lon"] != 13:
        raise ValueError(
            f"A grade do artigo deve ser 13x13; recebida {data.sizes['lat']}x{data.sizes['lon']}"
        )
    if data.sizes["time"] != 216:
        raise ValueError(f"O período 1998-2015 deve conter 216 meses; recebidos {data.sizes['time']}")

    actual = pd.DatetimeIndex(data.time.values).to_period("M")
    expected = pd.period_range("1998-01", "2015-12", freq="M")
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise ValueError(f"Eixo mensal inconsistente. Ausentes={list(missing)}, extras={list(extra)}")
    if not np.isfinite(data.values).all():
        raise ValueError("Há valores de precipitação ausentes ou não finitos.")
    if (data.values < 0).any():
        raise ValueError("Precipitação mensal não pode ser negativa.")


def save_compact_netcdf(data: xr.DataArray, path: str | Path) -> None:
    """Salva a pequena grade mensal em um artefato local compacto e autocontido."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # O backend scipy lida de forma confiável com caminhos Unicode no Windows.
    data.to_netcdf(target, engine="scipy")
