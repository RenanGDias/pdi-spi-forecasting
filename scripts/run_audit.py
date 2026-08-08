"""Executa reprodução paper-like, auditoria e protocolo causal corrigido."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spi_audit.data import load_monthly_trmm_csv, save_compact_netcdf
from spi_audit.datasets import (
    annual_origin_indices,
    make_causal_annual_design,
    make_paper_like_design,
    sequential_indices,
)
from spi_audit.deep import train_recursive_spatial_cnn
from spi_audit.evaluation import metrics_by_month, metrics_by_split
from spi_audit.metrics import regression_metrics
from spi_audit.models import candidate_regressors, paper_mlp
from spi_audit.overlap import overlap_table
from spi_audit.spi import compute_spi


SCALES = (1, 3, 6, 9, 12, 18, 24, 48)


def monthly_multioutput_metrics(observed, predicted, scale, model, protocol):
    records = []
    for horizon in range(observed.shape[1]):
        record = {
            "scale": scale,
            "model": model,
            "protocol": protocol,
            "month": horizon + 1,
            "horizon_months": horizon + 1,
        }
        record.update(regression_metrics(observed[:, horizon], predicted[:, horizon]))
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="CSV mensal exportado pelo Earth Engine")
    parser.add_argument("--output", type=Path, default=Path("outputs/resultados"))
    parser.add_argument("--all-paper-models", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--scales", type=int, nargs="+", default=list(SCALES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    precipitation = load_monthly_trmm_csv(args.csv)
    save_compact_netcdf(precipitation, args.output / "trmm_monthly.nc")
    overlap_table().to_csv(args.output / "spi_window_overlap.csv", index=False)

    paper_splits, paper_months, causal_metrics, selection = [], [], [], []
    deep_metrics = []
    year_counts = range(1, 18) if args.all_paper_models else (15,)

    invalid_scales = sorted(set(args.scales) - set(SCALES))
    if invalid_scales:
        raise ValueError(f"Escalas não suportadas: {invalid_scales}")

    for scale in args.scales:
        # Cenário deliberadamente permissivo para testar a interpretação do artigo:
        # calibra o SPI incluindo 2015 e aceita janelas iniciais parciais.
        paper_spi = compute_spi(
            precipitation,
            scale,
            calibration_end="2015-12-31",
            min_periods=1,
        )
        save_compact_netcdf(paper_spi, args.output / f"spi_{scale}_paper_global.nc")
        for year_count in year_counts:
            design = make_paper_like_design(paper_spi, year_count, target_year=2015)
            split = sequential_indices(len(design.y))
            model = paper_mlp(hidden_neurons=11, random_state=42 + year_count + scale)
            model.fit(design.X[split["train"]], design.y[split["train"]])
            prediction = model.predict(design.X)
            context = {"scale": scale, "input_years": year_count, "protocol": "paper_like"}
            paper_splits.append(metrics_by_split(design.y, prediction, split, context))
            if year_count == 15:
                paper_months.append(
                    metrics_by_month(design.y, prediction, design.metadata, context)
                )
            validation_score = regression_metrics(
                design.y[split["validation"]], prediction[split["validation"]]
            )
            test_score = regression_metrics(design.y[split["test"]], prediction[split["test"]])
            selection.append(
                {
                    "scale": scale,
                    "input_years": year_count,
                    "validation_rmse": validation_score["rmse"],
                    "test_rmse": test_score["rmse"],
                }
            )

        # Protocolo causal: SPI calibrado só até 2013, previsão fixa dos 12 meses.
        causal_spi = compute_spi(
            precipitation,
            scale,
            calibration_end="2013-12-31",
            min_periods=scale,
        )
        save_compact_netcdf(causal_spi, args.output / f"spi_{scale}_causal.nc")
        annual = make_causal_annual_design(causal_spi, lookback_months=12, forecast_months=12)
        origin_split = annual_origin_indices(annual.metadata)
        train, validation, test = (
            origin_split["train"],
            origin_split["validation"],
            origin_split["test"],
        )
        if min(len(train), len(validation), len(test)) == 0:
            raise RuntimeError(f"Partição causal vazia para SPI-{scale}")

        baseline_predictions = {
            "persistence_last": np.repeat(annual.X[test, -1:], 12, axis=1),
            "seasonal_persistence": annual.X[test, -12:],
        }
        for name, prediction in baseline_predictions.items():
            causal_metrics.extend(
                monthly_multioutput_metrics(annual.y[test], prediction, scale, name, "causal")
            )

        candidates = candidate_regressors(random_state=42 + scale)
        validation_results = {}
        fitted = {}
        for name, model in candidates.items():
            model.fit(annual.X[train], annual.y[train])
            validation_prediction = model.predict(annual.X[validation])
            validation_results[name] = regression_metrics(
                annual.y[validation], validation_prediction
            )["rmse"]
            fitted[name] = model
        selected = min(validation_results, key=validation_results.get)
        for name, model in fitted.items():
            prediction = model.predict(annual.X[test])
            records = monthly_multioutput_metrics(
                annual.y[test], prediction, scale, name, "causal"
            )
            for record in records:
                record["selected_by_validation"] = name == selected
                record["validation_rmse"] = validation_results[name]
            causal_metrics.extend(records)

        if args.deep:
            result = train_recursive_spatial_cnn(causal_spi)
            observed = result.observed.reshape(12, -1).T
            predicted = result.predicted.reshape(12, -1).T
            records = monthly_multioutput_metrics(
                observed, predicted, scale, "spatial_cnn_recursive", "causal_pdi"
            )
            for record in records:
                record["validation_loss"] = result.best_validation_loss
                record["epochs_trained"] = result.epochs_trained
            deep_metrics.extend(records)

    pd.concat(paper_splits, ignore_index=True).to_csv(
        args.output / "paper_like_metrics_by_split.csv", index=False
    )
    pd.concat(paper_months, ignore_index=True).to_csv(
        args.output / "paper_like_ann15_by_month.csv", index=False
    )
    pd.DataFrame(selection).to_csv(args.output / "paper_like_model_selection.csv", index=False)
    pd.DataFrame(causal_metrics).to_csv(args.output / "causal_metrics.csv", index=False)
    if deep_metrics:
        pd.DataFrame(deep_metrics).to_csv(args.output / "causal_pdi_cnn_metrics.csv", index=False)

    summary = {
        "data": str(args.csv),
        "scales": list(args.scales),
        "paper_year_counts": list(year_counts),
        "deep_learning_executed": args.deep,
        "output": str(args.output),
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
