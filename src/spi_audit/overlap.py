"""Quantificação da parcela já conhecida nas janelas acumuladas de SPI."""

from __future__ import annotations

import pandas as pd


def known_window_fraction(target: str | pd.Timestamp, scale: int, origin: str | pd.Timestamp) -> dict:
    target_period = pd.Timestamp(target).to_period("M")
    origin_period = pd.Timestamp(origin).to_period("M")
    window = pd.period_range(target_period - (scale - 1), target_period, freq="M")
    known = int((window <= origin_period).sum())
    return {
        "target": target_period.to_timestamp(),
        "scale": scale,
        "known_months": known,
        "unknown_months": scale - known,
        "known_fraction": known / scale,
    }


def overlap_table(
    scales: list[int] | tuple[int, ...] = (1, 3, 6, 9, 12, 18, 24, 48),
    origin: str = "2014-12-31",
    target_year: int = 2015,
) -> pd.DataFrame:
    records = []
    for month in range(1, 13):
        target = pd.Timestamp(year=target_year, month=month, day=1)
        records.extend(known_window_fraction(target, scale, origin) for scale in scales)
    return pd.DataFrame.from_records(records)


def annual_lag_overlap(scale: int, lag_years: int = 1) -> float:
    """Sobreposição entre a janela do alvo e a do mesmo mês ``lag_years`` antes.

    Esta é a sobreposição que importa no desenho do artigo. A rede não recebe
    os meses que antecedem a origem: ela recebe o **mesmo mês** em anos
    anteriores. Duas janelas de ``scale`` meses separadas por doze meses
    compartilham ``max(0, scale - 12 * lag_years)`` meses, de modo que escalas
    de até doze meses não têm sobreposição alguma.

    Não confundir com :func:`known_window_fraction`, que responde a outra
    pergunta: quanto da janela do alvo já é conhecido por um previsor
    posicionado em uma data de origem.
    """
    if scale < 1:
        raise ValueError("A escala do SPI deve ser positiva.")
    if lag_years < 1:
        raise ValueError("lag_years deve ser positivo.")
    return max(0, scale - 12 * lag_years) / scale


def annual_lag_overlap_table(
    scales: list[int] | tuple[int, ...] = (1, 3, 6, 9, 12, 18, 24, 48),
    lag_years: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "scale": scale,
                "lag_years": lag_years,
                "shared_months": max(0, scale - 12 * lag_years),
                "overlap_fraction": annual_lag_overlap(scale, lag_years),
            }
            for scale in scales
        ]
    )
