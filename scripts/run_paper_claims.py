"""Testa diretamente as três afirmações centrais do artigo.

Os experimentos de `run_audit.py` assumem a ANN#15 como dada e avaliam o
protocolo. Aqui as próprias afirmações do artigo são colocadas à prova:

A. **A ANN#15 é a melhor entre as seções ANN#1..ANN#17?**
   O artigo escolhe a ANN#15 comparando 17 modelos pelo desempenho das
   previsões de 2015. Reproduzindo essa comparação, a métrica cresce de forma
   monótona com o número de anos de entrada, porque ela é calculada sobre os
   169 pontos da grade e 70% deles estão no treino. O que a curva mede é
   capacidade de ajuste, não habilidade de previsão.

B. **As conclusões da auditoria dependem da ausência de early stopping?**
   O artigo declara parada por MSE de validação, que o L-BFGS não expõe. O
   contraste é refeito com `paper_mlp(early_stopping=True)` para verificar se
   a evidência de vazamento sobrevive.

C. **A sobreposição das janelas explica os valores publicados?**
   No desenho do artigo a entrada é o mesmo mês em anos anteriores, então a
   sobreposição relevante é `max(0, escala - 12) / escala`, e não a fração já
   conhecida em uma origem. A persistência trivial (copiar o SPI de 2014 para
   2015) mede quanta correlação essa sobreposição sozinha produz.

Requer os arquivos `spi_*_paper_global.nc` já gravados por `run_audit.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from spi_audit.datasets import make_paper_like_design, sequential_indices
from spi_audit.metrics import regression_metrics
from spi_audit.models import paper_mlp
from spi_audit.overlap import annual_lag_overlap

SCALES = (1, 3, 6, 9, 12, 18, 24, 48)
MAX_INPUT_YEARS = 17


def _point_major(design):
    """Ordena as amostras por ponto da grade, como em `analyze_paper_orderings`."""
    order = np.lexsort(
        (design.metadata["month"].to_numpy(), design.metadata["point_id"].to_numpy())
    )
    return design.X[order], design.y[order], design.metadata.iloc[order].reset_index(drop=True)


def _monthly_metric(observed, predicted, metadata, key="r"):
    """Média entre os doze meses da métrica calculada sobre os pontos da grade.

    É a forma das Tabelas 2 e 3: um valor por mês, agregando pontos.
    """
    values = [
        regression_metrics(observed[metadata.month == month], predicted[metadata.month == month])[key]
        for month in range(1, 13)
    ]
    return float(np.nanmean(values))


def _load_reference(path: Path) -> dict[int, np.ndarray]:
    reported = pd.read_csv(path)
    return {
        int(column.removeprefix("SPI-")): reported[column].to_numpy()
        for column in reported.columns
        if column != "month"
    }


def _monthly_series(observed, predicted, metadata, key="r"):
    return np.array(
        [
            regression_metrics(
                observed[metadata.month == month], predicted[metadata.month == month]
            )[key]
            for month in range(1, 13)
        ]
    )


def experiment_model_sweep(spi_by_scale: dict[int, xr.DataArray]) -> pd.DataFrame:
    """A: reproduz a comparação ANN#1..ANN#17 feita pelo artigo."""
    records = []
    for scale, spi in spi_by_scale.items():
        for input_years in range(1, MAX_INPUT_YEARS + 1):
            design = make_paper_like_design(spi, input_years, target_year=2015)
            X, y, metadata = _point_major(design)
            split = sequential_indices(len(y))
            model = paper_mlp(random_state=57 + scale + input_years)
            model.fit(X[split["train"]], y[split["train"]])
            prediction = model.predict(X)

            test = split["test"]
            test_metadata = metadata.iloc[test].reset_index(drop=True)
            records.append(
                {
                    "scale": scale,
                    "input_years": input_years,
                    "r_all_points": _monthly_metric(y, prediction, metadata),
                    "kappa_all_points": _monthly_metric(y, prediction, metadata, "kappa"),
                    "r_internal_test": _monthly_metric(
                        y[test], prediction[test], test_metadata
                    ),
                }
            )
        print(f"varredura ANN#1..#{MAX_INPUT_YEARS} concluída para SPI-{scale}", flush=True)
    return pd.DataFrame(records)


def experiment_early_stopping(
    spi_by_scale: dict[int, xr.DataArray], reference: dict[int, np.ndarray]
) -> pd.DataFrame:
    """B: refaz o contraste artigo x todos x teste com e sem early stopping."""
    records = []
    for scale, spi in spi_by_scale.items():
        design = make_paper_like_design(spi, 15, target_year=2015)
        X, y, metadata = _point_major(design)
        split = sequential_indices(len(y))
        for early_stopping in (False, True):
            model = paper_mlp(random_state=57 + scale, early_stopping=early_stopping)
            model.fit(X[split["train"]], y[split["train"]])
            prediction = model.predict(X)

            test = split["test"]
            test_metadata = metadata.iloc[test].reset_index(drop=True)
            train = split["train"]
            train_metadata = metadata.iloc[train].reset_index(drop=True)

            monthly_all = _monthly_series(y, prediction, metadata)
            monthly_test = _monthly_series(y[test], prediction[test], test_metadata)
            error_all = float(np.nanmean(np.abs(reference[scale] - monthly_all)))
            error_test = float(np.nanmean(np.abs(reference[scale] - monthly_test)))
            records.append(
                {
                    "scale": scale,
                    "early_stopping": early_stopping,
                    "r_train": _monthly_metric(y[train], prediction[train], train_metadata),
                    "r_all_points": float(np.nanmean(monthly_all)),
                    "r_internal_test": float(np.nanmean(monthly_test)),
                    "abs_error_vs_all": error_all,
                    "abs_error_vs_test": error_test,
                    "closer_to_all": error_all < error_test,
                }
            )
        print(f"contraste de early stopping concluído para SPI-{scale}", flush=True)
    return pd.DataFrame(records)


def experiment_persistence(
    spi_by_scale: dict[int, xr.DataArray], reference: dict[int, np.ndarray]
) -> pd.DataFrame:
    """C: persistência trivial contra sobreposição de janelas e R publicado."""
    records = []
    for scale, spi in spi_by_scale.items():
        design = make_paper_like_design(spi, 15, target_year=2015)
        X, y, metadata = _point_major(design)
        # O último atributo é o mesmo mês em 2014: previsão de persistência anual.
        persistence = X[:, -1]
        records.append(
            {
                "scale": scale,
                "window_overlap": annual_lag_overlap(scale),
                "r_persistence": _monthly_metric(y, persistence, metadata),
                "kappa_persistence": _monthly_metric(y, persistence, metadata, "kappa"),
                "r_reported": float(np.nanmean(reference[scale])),
            }
        )
    frame = pd.DataFrame(records)
    frame["unexplained_by_overlap"] = frame["r_reported"] - frame["r_persistence"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Pasta com os spi_*_paper_global.nc")
    parser.add_argument("--reference", type=Path, default=Path("reference/article_table2_r.csv"))
    parser.add_argument("--scales", type=int, nargs="+", default=list(SCALES))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    reference = _load_reference(args.reference)
    # O backend scipy lida corretamente com o caminho Unicode no Windows.
    spi_by_scale = {
        scale: xr.load_dataarray(args.results / f"spi_{scale}_paper_global.nc", engine="scipy")
        for scale in args.scales
    }

    sweep = experiment_model_sweep(spi_by_scale)
    sweep.to_csv(args.results / "paper_claim_model_sweep.csv", index=False)
    contrast = experiment_early_stopping(spi_by_scale, reference)
    contrast.to_csv(args.results / "paper_claim_early_stopping.csv", index=False)
    persistence = experiment_persistence(spi_by_scale, reference)
    persistence.to_csv(args.results / "paper_claim_persistence.csv", index=False)

    table = sweep.pivot(index="input_years", columns="scale", values="r_all_points")
    print("\n=== A. R médio mensal sobre os 169 pontos, por ANN#k ===")
    print(table.round(3).to_string())
    mean_by_model = table.mean(axis=1)
    print("\nmédia entre escalas:")
    print(mean_by_model.round(3).to_string())
    print(f"melhor ANN pela métrica do artigo: ANN#{int(mean_by_model.idxmax())}")
    monotonic = bool((mean_by_model.diff().dropna() > 0).all())
    print(f"a métrica cresce monotonicamente com os anos de entrada: {monotonic}")

    print("\n=== B. Contraste com e sem early stopping ===")
    print(contrast.round(3).to_string(index=False))
    for early_stopping in (False, True):
        subset = contrast[contrast.early_stopping == early_stopping]
        print(
            f"early_stopping={early_stopping}: erro vs todos={subset.abs_error_vs_all.mean():.3f}"
            f" | erro vs teste={subset.abs_error_vs_test.mean():.3f}"
            f" | mais perto de todos em {int(subset.closer_to_all.sum())}/{len(subset)} escalas"
        )

    print("\n=== C. Persistência trivial e sobreposição de janelas ===")
    print(persistence.round(3).to_string(index=False))
    print(
        f"média: persistência={persistence.r_persistence.mean():.3f}"
        f" | publicado={persistence.r_reported.mean():.3f}"
    )


if __name__ == "__main__":
    main()
