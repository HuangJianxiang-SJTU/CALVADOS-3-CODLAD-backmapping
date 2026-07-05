#!/usr/bin/env python3
"""Analyze how Fig. 2 clash counts depend on chain length and compactness."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT_FIG = ROOT / "manuscript" / "figures" / "clash_length_compactness_rebuttal"
OUT_TABLE = ROOT / "manuscript" / "tables" / "clash_length_compactness_rebuttal.csv"
OUT_STATS = ROOT / "manuscript" / "tables" / "clash_length_compactness_stats.csv"

if OUT_TABLE.exists():
    # The committed table is the self-contained input used for the rebuttal figure.
    # Regenerating it from topology cache files requires the large untracked fig2 cache.
    df = pd.read_csv(OUT_TABLE)
else:
    raise FileNotFoundError(
        f"Missing {OUT_TABLE}. This plotting script is distributed with the precomputed "
        "clash-length table so the figure can be regenerated without the large fig2 topology cache."
    )

pairs = [
    ("clash_pred_mean", "n_residues"),
    ("clash_pred_per_residue", "n_residues"),
    ("clash_delta", "n_residues"),
    ("clash_pred_mean", "compactness_scaled"),
    ("clash_pred_per_residue", "compactness_scaled"),
    ("clash_delta", "compactness_scaled"),
]
stat_rows = []
for y, x in pairs:
    pr = pearsonr(df[x], df[y])
    sr = spearmanr(df[x], df[y])
    stat_rows.append({
        "x": x,
        "y": y,
        "pearson_r": pr.statistic,
        "pearson_p": pr.pvalue,
        "spearman_rho": sr.statistic,
        "spearman_p": sr.pvalue,
    })
pd.DataFrame(stat_rows).to_csv(OUT_STATS, index=False)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
plots = [
    ("n_residues", "clash_pred_mean", "Residues", "Predicted clashes", "A"),
    ("n_residues", "clash_pred_per_residue", "Residues", "Predicted clashes / residue", "B"),
    ("compactness_scaled", "clash_delta", "Compactness, 1/(Rg/N^0.6)", "CODLAD - PED clashes", "C"),
]
for ax, (x, y, xlabel, ylabel, panel) in zip(axes, plots):
    ax.scatter(df[x], df[y], s=38, color="#3B6FB6", edgecolor="white", linewidth=0.7, zorder=3)
    coef = np.polyfit(df[x], df[y], deg=1)
    xx = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color="#333333", lw=1.4)
    sr = spearmanr(df[x], df[y])
    ax.text(0.04, 0.94, f"rho={sr.statistic:.2f}\np={sr.pvalue:.3g}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(-0.18, 1.08, panel, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
fig.suptitle("Dependence of Fig. 2 clash counts on chain length and compactness", fontsize=11)
for ext in ["png", "svg"]:
    fig.savefig(f"{OUT_FIG}.{ext}", dpi=300)
print(f"Wrote {OUT_TABLE}")
print(f"Wrote {OUT_STATS}")
print(f"Wrote {OUT_FIG}.png/.svg")
print(pd.read_csv(OUT_STATS).round(4).to_string(index=False))
