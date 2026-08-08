import numpy as np
import pandas as pd
import xarray as xr

from spi_audit.datasets import (
    annual_origin_indices,
    make_causal_annual_design,
    make_paper_like_design,
    sequential_indices,
)


def synthetic_spi(start="1998-01-01", end="2015-12-01"):
    times = pd.date_range(start, end, freq="MS")
    lat = [-20.0, -19.75]
    lon = [-45.0, -44.75]
    rng = np.random.default_rng(7)
    values = rng.normal(size=(len(times), len(lat), len(lon)))
    return xr.DataArray(values, coords={"time": times, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"))


def test_paper_like_design_uses_years_as_features():
    design = make_paper_like_design(synthetic_spi(), input_year_count=15, target_year=2015)
    assert design.X.shape == (12 * 4, 15)
    assert design.y.shape == (12 * 4,)
    assert design.metadata.target_time.nunique() == 12
    split = sequential_indices(len(design.y))
    assert sum(len(indices) for indices in split.values()) == len(design.y)


def test_annual_design_has_strict_origins():
    design = make_causal_annual_design(synthetic_spi(), 12, 12)
    split = annual_origin_indices(design.metadata)
    assert len(split["validation"]) == 4
    assert len(split["test"]) == 4
    assert (design.metadata.iloc[split["test"]].origin_time == pd.Timestamp("2014-12-01")).all()
    assert (
        design.metadata.iloc[split["train"]].target_end <= pd.Timestamp("2013-12-31")
    ).all()

