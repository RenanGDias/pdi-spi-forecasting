from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from spi_audit.deep import train_recursive_spatial_cnn
from spi_audit.evaluation import metrics_by_month
from spi_audit.metrics import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa apenas a CNN espacial causal.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 3, 6, 9, 12, 18, 24, 48])
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    monthly_frames: list[pd.DataFrame] = []
    summary_records: list[dict] = []

    for scale in args.scales:
        spi = xr.load_dataarray(args.results / f"spi_{scale}_causal.nc", engine="scipy")
        result = train_recursive_spatial_cnn(spi, random_state=420 + scale)

        observed = result.observed.reshape(12, -1).T
        predicted = result.predicted.reshape(12, -1).T
        metadata = pd.DataFrame({"target_time": result.target_times})
        # metrics_by_month espera uma linha por amostra. Repetimos cada mês por ponto.
        point_count = observed.shape[0]
        long_observed = observed.T.reshape(-1)
        long_predicted = predicted.T.reshape(-1)
        long_metadata = pd.DataFrame(
            {"target_time": result.target_times.repeat(point_count)}
        )
        monthly = metrics_by_month(
            long_observed,
            long_predicted,
            long_metadata,
            extra={
                "scale": scale,
                "model": "spatial_residual_cnn",
                "protocol": "causal_pdi_recursive",
                "validation_loss": result.best_validation_loss,
                "epochs_trained": result.epochs_trained,
            },
        )
        monthly_frames.append(monthly)

        overall = regression_metrics(observed, predicted)
        overall.update(
            {
                "scale": scale,
                "model": "spatial_residual_cnn",
                "validation_loss": result.best_validation_loss,
                "epochs_trained": result.epochs_trained,
            }
        )
        summary_records.append(overall)

        prediction_array = xr.DataArray(
            result.predicted,
            dims=("time", "lat", "lon"),
            coords={"time": result.target_times, "lat": spi.lat, "lon": spi.lon},
            name=f"spi_{scale}_prediction",
            attrs={
                "protocol": "train through 2013; validation 2014; recursive test 2015",
                "model": "spatial residual CNN with gradient loss",
            },
        )
        prediction_array.to_netcdf(
            args.results / f"spi_{scale}_cnn_prediction_2015.nc", engine="scipy"
        )

    pd.concat(monthly_frames, ignore_index=True).to_csv(
        args.results / "causal_pdi_cnn_metrics.csv", index=False
    )
    summary = pd.DataFrame(summary_records).sort_values("scale")
    summary.to_csv(args.results / "causal_pdi_cnn_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
