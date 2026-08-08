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
