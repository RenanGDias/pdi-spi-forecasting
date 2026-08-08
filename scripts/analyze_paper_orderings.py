from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from spi_audit.datasets import make_paper_like_design, sequential_indices
from spi_audit.evaluation import metrics_by_month, metrics_by_split
from spi_audit.models import paper_mlp


SCALES = (1, 3, 6, 9, 12, 18, 24, 48)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testa a divisão 70/15/15 com amostras ordenadas por ponto espacial."
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("reference/article_table2_r.csv"))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    split_frames: list[pd.DataFrame] = []
    all_month_frames: list[pd.DataFrame] = []
    test_month_frames: list[pd.DataFrame] = []
    partition_records: list[dict] = []

    for scale in SCALES:
        # O backend scipy lida corretamente com o caminho Unicode no Windows.
        spi = xr.load_dataarray(
            args.results / f"spi_{scale}_paper_global.nc", engine="scipy"
        )
        design = make_paper_like_design(spi, input_year_count=15)

        # Última chave é a primária em np.lexsort: ponto primeiro, mês depois.
        order = np.lexsort(
            (design.metadata["month"].to_numpy(), design.metadata["point_id"].to_numpy())
        )
        X = design.X[order]
        y = design.y[order]
        metadata = design.metadata.iloc[order].reset_index(drop=True)
        split = sequential_indices(len(y))

        model = paper_mlp(hidden_neurons=11, random_state=57 + scale)
        model.fit(X[split["train"]], y[split["train"]])
        prediction = model.predict(X)

        extra = {"scale": scale, "input_years": 15, "ordering": "point_major"}
        split_frames.append(metrics_by_split(y, prediction, split, extra=extra))
        all_month_frames.append(metrics_by_month(y, prediction, metadata, extra=extra))

        test_index = split["test"]
        test_month_frames.append(
            metrics_by_month(
                y[test_index],
                prediction[test_index],
                metadata.iloc[test_index].reset_index(drop=True),
                extra={**extra, "subset": "test"},
            )
        )

        for split_name, indices in split.items():
            subset = metadata.iloc[indices]
            partition_records.append(
                {
                    "scale": scale,
                    "split": split_name,
                    "n": len(indices),
                    "unique_months": subset["month"].nunique(),
                    "unique_points": subset["point_id"].nunique(),
                    "point_min": subset["point_id"].min(),
                    "point_max": subset["point_id"].max(),
                }
            )

    split_metrics = pd.concat(split_frames, ignore_index=True)
    monthly_all = pd.concat(all_month_frames, ignore_index=True)
    monthly_test = pd.concat(test_month_frames, ignore_index=True)
    partitions = pd.DataFrame(partition_records)

    split_metrics.to_csv(args.results / "paper_like_point_major_metrics_by_split.csv", index=False)
    monthly_all.to_csv(args.results / "paper_like_point_major_by_month_all.csv", index=False)
    monthly_test.to_csv(args.results / "paper_like_point_major_by_month_test.csv", index=False)
    partitions.to_csv(args.results / "paper_like_point_major_partitions.csv", index=False)

    reported = pd.read_csv(args.reference).rename(columns={"month": "month_name"})
    month_number = {name: index for index, name in enumerate(reported["month_name"], start=1)}
    reported_long = reported.melt(
        id_vars="month_name", var_name="scale_name", value_name="reported_r"
    )
    reported_long["month"] = reported_long["month_name"].map(month_number)
    reported_long["scale"] = reported_long["scale_name"].str.removeprefix("SPI-").astype(int)

    comparison = reported_long.merge(
        monthly_all[["scale", "month", "r"]].rename(columns={"r": "point_major_all_r"}),
        on=["scale", "month"],
    ).merge(
        monthly_test[["scale", "month", "r"]].rename(columns={"r": "point_major_test_r"}),
        on=["scale", "month"],
    )
    comparison["abs_error_all"] = (
        comparison["reported_r"] - comparison["point_major_all_r"]
    ).abs()
    comparison["abs_error_test"] = (
        comparison["reported_r"] - comparison["point_major_test_r"]
    ).abs()
    comparison.to_csv(args.results / "reported_vs_point_major_monthly_r.csv", index=False)

    summary = comparison.groupby("scale")[["abs_error_all", "abs_error_test"]].mean()
    summary.to_csv(args.results / "reported_vs_point_major_summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
