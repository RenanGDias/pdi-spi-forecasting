import numpy as np
import pandas as pd
import xarray as xr

from spi_audit.spi import compute_spi


def precipitation_fixture():
    times = pd.date_range("1998-01-01", "2015-12-01", freq="MS")
    lat = [-20.0, -19.75]
    lon = [-45.0, -44.75]
    month = np.arange(len(times)) % 12
    seasonal = 100 + 60 * np.sin(2 * np.pi * month / 12)
    rng = np.random.default_rng(11)
    values = seasonal[:, None, None] + rng.gamma(2, 8, size=(len(times), 2, 2))
    return xr.DataArray(
        values,
        coords={"time": times, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="precipitation_mm",
    )


def test_spi_is_finite_after_full_accumulation_window():
    spi = compute_spi(precipitation_fixture(), 3, calibration_end="2013-12-31")
    assert np.isnan(spi.values[:2]).all()
    assert np.isfinite(spi.values[2:]).all()
    assert spi.attrs["calibration_end"] == "2013-12-31"


def test_future_changes_do_not_change_training_spi_with_causal_calibration():
    original = precipitation_fixture()
    modified = original.copy(deep=True)
    modified.loc[{"time": slice("2014-01-01", "2015-12-01")}] *= 10
    first = compute_spi(original, 1, calibration_end="2013-12-31")
    second = compute_spi(modified, 1, calibration_end="2013-12-31")
    np.testing.assert_allclose(
        first.sel(time=slice(None, "2013-12-01")),
        second.sel(time=slice(None, "2013-12-01")),
    )

