"""Relaciona as correlações publicadas à fração conhecida das janelas de SPI."""

import os
from pathlib import Path

matplotlib_config = Path("tmp/matplotlib").resolve()
matplotlib_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spi_audit.overlap import overlap_table


output = Path("outputs/resultados")
output.mkdir(parents=True, exist_ok=True)

reported = pd.read_csv("reference/article_table2_r.csv").melt(
    "month", var_name="scale_name", value_name="reported_r"
)
reported["scale"] = reported.scale_name.str.extract(r"(\d+)").astype(int)
reported["month_number"] = pd.to_datetime(reported.month, format="%B").dt.month

overlap = overlap_table()
overlap["month_number"] = overlap.target.dt.month
analysis = reported.merge(
    overlap[["scale", "month_number", "known_months", "unknown_months", "known_fraction"]],
    on=["scale", "month_number"],
)
analysis["escala"] = "SPI-" + analysis["scale"].astype(str)
analysis.to_csv(output / "reported_r_vs_overlap.csv", index=False)

sns.set_theme(style="whitegrid")
figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
sns.scatterplot(
    data=analysis,
    x="known_fraction",
    y="reported_r",
    hue="escala",
    palette="viridis",
    s=72,
    ax=axis,
)
sns.regplot(
    data=analysis,
    x="known_fraction",
    y="reported_r",
    scatter=False,
    color="black",
    line_kws={"linewidth": 1.5, "linestyle": "--"},
    ax=axis,
)
correlation = analysis.reported_r.corr(analysis.known_fraction)
axis.set(
    xlabel="Fração da janela de SPI anterior à origem (dez/2014)",
    ylabel="Correlação R publicada para 2015",
    title=f"Desempenho publicado versus meses já conhecidos (r = {correlation:.3f})",
    xlim=(-0.03, 1.03),
    ylim=(-0.08, 1.03),
)
axis.legend(title="Escala", loc="lower right", frameon=True)
figure.savefig(output / "reported_r_vs_overlap.png", dpi=180)
print(f"correlation={correlation:.6f}")
