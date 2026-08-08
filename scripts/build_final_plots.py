from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera as figuras finais da auditoria.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results

    sns.set_theme(style="whitegrid", context="talk")

    comparison = pd.read_csv(out / "reported_vs_point_major_monthly_r.csv")
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), sharex=True, sharey=True)
    for axis, scale in zip(axes.flat, sorted(comparison["scale"].unique())):
        subset = comparison[comparison["scale"] == scale].sort_values("month")
        axis.plot(subset["month"], subset["reported_r"], "o-k", label="Artigo")
        axis.plot(
            subset["month"],
            subset["point_major_all_r"],
            "o-",
            color="#d55e00",
            label="Todos (inclui treino)",
        )
        axis.plot(
            subset["month"],
            subset["point_major_test_r"],
            "o--",
            color="#0072b2",
            label="Teste interno",
        )
        axis.axhline(0, color="0.55", linewidth=0.8)
        axis.set_title(f"SPI-{scale}")
        axis.set_xticks([1, 3, 6, 9, 12])
        axis.set_ylim(-1.05, 1.05)
    axes[0, 0].legend(loc="lower left", fontsize=10)
    fig.supxlabel("Mês de 2015")
    fig.supylabel("Correlação de Pearson (r)")
    fig.suptitle("Métricas publicadas se aproximam da avaliação que inclui amostras de treino")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    fig.savefig(out / "leakage_monthly_r_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    causal = pd.read_csv(out / "causal_metrics.csv")
    cnn = pd.read_csv(out / "causal_pdi_cnn_metrics.csv")
    selected = (
        causal[causal["selected_by_validation"].fillna(False)]
        .groupby(["scale", "model"], as_index=False)[["rmse", "mae", "r", "kappa"]]
        .mean()
    )
    selected["series"] = "Regressor escolhido em 2014"
    persistence = (
        causal[causal["model"].isin(["persistence_last", "seasonal_persistence"])]
        .groupby(["scale", "model"], as_index=False)[["rmse", "mae", "r", "kappa"]]
        .mean()
    )
    persistence["series"] = persistence["model"].map(
        {
            "persistence_last": "Persistência do último SPI",
            "seasonal_persistence": "Persistência sazonal",
        }
    )
    cnn_summary = cnn.groupby(["scale", "model"], as_index=False)[
        ["rmse", "mae", "r", "kappa"]
    ].mean()
    cnn_summary["series"] = "CNN espacial causal + PDI"
    plot_data = pd.concat([selected, persistence, cnn_summary], ignore_index=True)

    fig, axis = plt.subplots(figsize=(14, 7))
    palette = {
        "Regressor escolhido em 2014": "#cc79a7",
        "Persistência do último SPI": "#009e73",
        "Persistência sazonal": "#e69f00",
        "CNN espacial causal + PDI": "#0072b2",
    }
    for name, subset in plot_data.groupby("series", sort=False):
        subset = subset.sort_values("scale")
        axis.plot(
            subset["scale"].astype(str),
            subset["rmse"],
            marker="o",
            linewidth=2.3,
            label=name,
            color=palette[name],
        )
    axis.set_xlabel("Escala do SPI")
    axis.set_ylabel("RMSE médio mensal em 2015 (menor é melhor)")
    axis.set_title("Comparação causal: 2015 nunca entra no ajuste nem na seleção")
    axis.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "causal_rmse_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    overlap = pd.read_csv(out / "spi_window_overlap.csv")
    heatmap = overlap.pivot(index="scale", columns="target", values="known_fraction")
    fig, axis = plt.subplots(figsize=(15, 6))
    sns.heatmap(
        heatmap,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Fração da janela SPI já conhecida em 31/12/2014"},
        ax=axis,
    )
    axis.set_xlabel("Mês-alvo de 2015")
    axis.set_ylabel("Escala do SPI")
    axis.set_title("Sobreposição das janelas explica a forte persistência em escalas longas")
    axis.set_xticklabels([str(value)[5:7] for value in heatmap.columns], rotation=0)
    fig.tight_layout()
    fig.savefig(out / "spi_overlap_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    evidence = (
        comparison.groupby("scale", as_index=False)[["abs_error_all", "abs_error_test"]]
        .mean()
        .rename(
            columns={
                "abs_error_all": "mae_r_artigo_vs_todos",
                "abs_error_test": "mae_r_artigo_vs_teste",
            }
        )
    )
    evidence["mais_proximo_de_todos"] = (
        evidence["mae_r_artigo_vs_todos"] < evidence["mae_r_artigo_vs_teste"]
    )
    evidence.to_csv(out / "leakage_evidence_summary.csv", index=False)


if __name__ == "__main__":
    main()
